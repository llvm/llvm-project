//===- LoopVersioning.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass looks for loops iterating over assumed-shape arrays, that can
/// be optimized by "guessing" that the stride is element-sized.
///
/// This is done by creating two versions of the same loop: one which assumes
/// that the elements are contiguous (stride == size of element), and one that
/// is the original generic loop.
///
/// As a side-effect of the assumed element size stride, slice-free accesses are
/// flattened to make the array 1D. This is required because the internal array
/// structure must be either 1D or have known sizes in all dimensions, and at
/// least one dimension here is already unknown. Sliced accesses instead use a
/// byte-address base and retain descriptor byte strides.
///
/// There are two distinct benefits here:
/// 1. The loop that iterates over the elements is somewhat simplified by the
///    constant stride calculation.
/// 2. Since the compiler can understand the size of the stride, it can use
///    vector instructions, where an unknown (at compile time) stride does often
///    prevent vector operations from being used.
///
/// A known drawback is that the code-size is increased, in some cases that can
/// be quite substantial - 3-4x is quite plausible (this includes that the loop
/// gets vectorized, which in itself often more than doubles the size of the
/// code, because unless the loop size is known, there will be a modulo
/// vector-size remainder to deal with.
///
/// TODO: Do we need some size limit where loops no longer get duplicated?
//        Maybe some sort of cost analysis.
/// TODO: Should some loop content - for example calls to functions and
///       subroutines inhibit the versioning of the loops. Plausibly, this
///       could be part of the cost analysis above.
//===----------------------------------------------------------------------===//

#include "flang/Common/ISO_Fortran_binding_wrapper.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace fir {
#define GEN_PASS_DEF_LOOPVERSIONING
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-loop-versioning"

namespace {

class LoopVersioningPass
    : public fir::impl::LoopVersioningBase<LoopVersioningPass> {
public:
  /// Construct the pass with its TableGen defaults.
  LoopVersioningPass() = default;
  /// Construct the pass with programmatic option values.
  LoopVersioningPass(fir::LoopVersioningOptions options) : Base(options) {}
  void runOnOperation() override;
};

/// Classifies one direct fir.slice triple using generic XArrayCoor semantics.
enum class SliceTripleKind { Scalar, Section, Unsupported };

/// Captures the lowering contract of one top-level FIR module.
/// Nested builtin modules are rejected before this record is constructed, so
/// one index width and kind mapping unambiguously govern later lowering.
struct SliceTargetInfo {
  unsigned indexWidth = 0;
  /// Kind mapping owned by runOnOperation and valid throughout preflight.
  const fir::KindMapping *kindMapping = nullptr;
};

/// Holds the effective source width and the low bits relevant to the 64-bit
/// address domain. Limiting the retained APInt prevents pathological integer
/// widths from multiplying cache memory while preserving every bit that can
/// affect the final static-one classification. APInt is used even though at
/// most 64 bits are retained: a single-word APInt stores the value inline and
/// provides width-aware truncation and sign or zero extension without manual
/// masks or special handling for width 64.
struct StaticIntegerState {
  unsigned width;
  llvm::APInt bits;
};

/// Caches the constant state produced at every visited SSA value in a
/// conversion chain. A missing key denotes an unvisited value, while a null
/// optional records a value that cannot be evaluated. Analysis runs before
/// modification, so every slice can share intermediate results without
/// repeatedly walking common conversion-chain prefixes.
using StaticIntegerCache =
    llvm::DenseMap<mlir::Value, std::optional<StaticIntegerState>>;

/// Caches the effective width of an MLIR type for one frozen target contract.
/// Types are uniqued and the target remains immutable throughout preflight, so
/// every coordinate, lower bound, and step of that type can share the result.
using SliceWidthCache = llvm::DenseMap<mlir::Type, unsigned>;

/// Immutable arithmetic decisions for one accepted source dimension.
struct SliceDimensionFacts {
  /// Whether the slice triple retains a section or selects one scalar.
  SliceTripleKind kind;
  /// Whether a retained section has no source-lower adjustment.
  bool lowerIsOne;
};

/// Per-dimension arithmetic decisions shared by accesses using one fir.slice,
/// stored in ascending source-dimension order.
using SliceFacts = llvm::SmallVector<SliceDimensionFacts, 4>;

/// Couples one exact sliced access with the facts frozen for its rewrite.
/// Discovery fills source in deterministic IR order. Preflight fills facts
/// only after validating that source, and emission uses both values directly.
struct SliceAccessPlan {
  /// Original operation whose clone is replaced in the fast path.
  fir::ArrayCoorOp source;
  /// Index of immutable slice facts, present only after successful preflight.
  std::optional<unsigned> factsIndex;
};

struct UseNode;

/// @struct ArgInfo
/// A structure to hold an argument, the size of the argument and dimension
/// information.
struct ArgInfo {
  mlir::Value arg;
  size_t size;
  unsigned rank;
  fir::BoxDimsOp dims[CFI_MAX_RANK];
  /// Element-size constant used by sliced byte-address emission.
  mlir::Value elemSize;
  /// Accepted direct-use node, or null for the existing slice-free path.
  UseNode *sliceNode = nullptr;
  /// Deterministic ordinal assigned at the first direct access.
  size_t firstUseOrder = std::numeric_limits<size_t>::max();
};

/// Stores the direct uses of one concrete descriptor for one nearest loop.
/// Only nodes containing a slice outlive collection. Their operation handles
/// form the deterministic worklist consumed by preflight and clone rewriting.
struct UseNode {
  /// Immediate loop containing every recorded access.
  fir::DoLoopOp loop;
  /// Descriptor and type information shared by the accesses.
  ArgInfo info;
  /// Sliced access plans in deterministic IR discovery order.
  llvm::SmallVector<SliceAccessPlan, 2> accesses;
  /// Facts in first-use order, shared by plans that use the same fir.slice.
  llvm::SmallVector<SliceFacts, 2> sliceFacts;
};

/// Holds one provisional sliced descriptor while collecting a single loop.
/// The wrapper owns the node until descriptor-wide preflight either retains
/// or rejects it.
struct SliceUse {
  /// Sliced accesses and their shared descriptor information.
  std::unique_ptr<UseNode> node;
  /// Whether this sliced group must remain on the generic path.
  bool rejected = false;
};

/// Owns the lazily created sliced-use tables for one loop callback.
/// Slice-free loops never construct this state.
struct LoopSliceUses {
  /// Provisional descriptor groups in first sliced-access order.
  llvm::SmallVector<SliceUse, 4> uses;
  /// Maps a sliced descriptor to its provisional group.
  llvm::DenseMap<mlir::Value, size_t> indices;
};

/// Groups retained sliced nodes of one concrete descriptor.
/// Slice-free and rejected descriptors are tracked separately by one compact
/// set, so they do not allocate owner vectors.
struct DescriptorUses {
  /// Stable nodes for owners containing at least one sliced access.
  llvm::SmallVector<UseNode *, 2> sliced;
  /// Whether two recorded owners have an ancestor-descendant relationship.
  bool nested = false;
};

/// Owns all function-lifetime state used only by slice discovery.
/// The optional owner in runOnOperation keeps this state absent when slice
/// support is disabled and retains accepted nodes through transformation.
struct SliceDiscovery {
  /// Retained sliced owners grouped by concrete descriptor.
  llvm::DenseMap<mlir::Value, DescriptorUses> descriptors;
  /// Descriptors with a slice-free or locally rejected direct owner.
  llvm::SmallDenseSet<mlir::Value, 4> rejected;
  /// Stable storage for every viable sliced owner.
  llvm::SmallVector<std::unique_ptr<UseNode>, 0> nodes;
  /// Next deterministic first-sliced-access ordinal.
  size_t nextUseOrder = 0;
};

/// Record one directly nested sliced access while collecting a loop.
/// The returned index remains valid if later accesses grow sliceUses; a
/// returned UseNode remains stable because it is heap allocated. The final
/// result reports whether the descriptor was first seen in this owner. A
/// rejected group returns no result and does not retain another access.
static std::optional<std::tuple<size_t, UseNode *, bool>>
recordSliceUse(fir::DoLoopOp loop, mlir::Value descriptor,
               fir::ArrayCoorOp arrayCoor, ArgInfo &info,
               LoopSliceUses &sliceUses, size_t &nextUseOrder) {
  auto [foundUse, firstDirectUse] =
      sliceUses.indices.try_emplace(descriptor, sliceUses.uses.size());
  if (firstDirectUse) {
    SliceUse &use = sliceUses.uses.emplace_back();
    use.node = std::make_unique<UseNode>();
    use.node->loop = loop;
    use.node->info.firstUseOrder = nextUseOrder++;
  }

  SliceUse &use = sliceUses.uses[foundUse->second];
  if (use.rejected)
    return std::nullopt;
  info.firstUseOrder = use.node->info.firstUseOrder;
  use.node->accesses.push_back({arrayCoor, std::nullopt});
  return std::tuple{foundUse->second, use.node.get(), firstDirectUse};
}

/// @struct ArgsUsageInLoop
/// A structure providing information about the function arguments
/// usage by instructions whose nearest enclosing do_loop is the given loop.
struct ArgsUsageInLoop {
  /// Mapping between the memref operand of an array indexing
  /// operation (e.g. fir.coordinate_of) and the argument information.
  llvm::DenseMap<mlir::Value, ArgInfo> usageInfo;
  /// Some array indexing operations inside a loop cannot be transformed.
  /// This vector holds the memref operands of such operations.
  /// The vector is used to make sure that we do not try to transform
  /// any outer loop, since this will imply the operation rewrite
  /// in this loop.
  llvm::SetVector<mlir::Value> cannotTransform;

  // Debug dump of the structure members assuming that
  // the information has been collected for the given loop.
  void dump(fir::DoLoopOp loop) const {
    LLVM_DEBUG({
      mlir::OpPrintingFlags printFlags;
      printFlags.skipRegions();
      llvm::dbgs() << "Arguments usage info for loop:\n";
      loop.print(llvm::dbgs(), printFlags);
      llvm::dbgs() << "\nUsed args:\n";
      bool hasSlices = llvm::any_of(
          usageInfo, [](const auto &use) { return use.second.sliceNode; });
      if (hasSlices) {
        assert(llvm::all_of(
                   usageInfo,
                   [](const auto &use) { return use.second.sliceNode; }) &&
               "a sliced owner must contain only sliced descriptors");
        llvm::SmallVector<const ArgInfo *, 4> args;
        for (const auto &use : usageInfo)
          args.push_back(&use.second);
        llvm::sort(args, [](const ArgInfo *left, const ArgInfo *right) {
          return left->firstUseOrder < right->firstUseOrder;
        });
        for (const ArgInfo *arg : args) {
          arg->arg.print(llvm::dbgs(), printFlags);
          llvm::dbgs() << "\n";
        }
      } else {
        for (auto &use : usageInfo) {
          mlir::Value v = use.first;
          v.print(llvm::dbgs(), printFlags);
          llvm::dbgs() << "\n";
        }
      }
      llvm::dbgs() << "\nCannot transform args:\n";
      for (mlir::Value arg : cannotTransform) {
        arg.print(llvm::dbgs(), printFlags);
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "====\n";
    });
  }

  // Erase usageInfo and cannotTransform entries for a set
  // of given arguments.
  void eraseUsage(const llvm::SetVector<mlir::Value> &args) {
    for (auto &arg : args)
      usageInfo.erase(arg);
    cannotTransform.set_subtract(args);
  }

  // Erase usageInfo and cannotTransform entries for a set
  // of given arguments provided in the form of usageInfo map.
  void eraseUsage(const llvm::DenseMap<mlir::Value, ArgInfo> &args) {
    for (auto &arg : args) {
      usageInfo.erase(arg.first);
      cannotTransform.remove(arg.first);
    }
  }
};

/// Maps each discovered loop to its descriptor-use summary.
using LoopUsageMap = llvm::DenseMap<fir::DoLoopOp, ArgsUsageInLoop>;
} // namespace

static fir::SequenceType getAsSequenceType(mlir::Value v) {
  mlir::Type argTy = fir::unwrapPassByRefType(fir::unwrapRefType(v.getType()));
  return mlir::dyn_cast<fir::SequenceType>(argTy);
}

/// Return the rank and the element size (in bytes) of the given
/// value \p v. If it is not an array or the element type is not
/// supported, then return <0, 0>. Only trivial data types
/// are currently supported.
/// When \p isArgument is true, \p v is assumed to be a function
/// argument. If \p v's type does not look like a type of an assumed
/// shape array, then the function returns <0, 0>.
/// When \p isArgument is false, array types with known innermost
/// dimension are allowed to proceed.
static std::pair<unsigned, size_t>
getRankAndElementSize(const fir::KindMapping &kindMap,
                      const mlir::DataLayout &dl, mlir::Value v,
                      bool isArgument = false) {
  if (auto seqTy = getAsSequenceType(v)) {
    unsigned rank = seqTy.getDimension();
    if (rank > 0 &&
        (!isArgument ||
         seqTy.getShape()[0] == fir::SequenceType::getUnknownExtent())) {
      size_t typeSize = 0;
      mlir::Type elementType = fir::unwrapSeqOrBoxedSeqType(v.getType());
      if (fir::isa_trivial(elementType)) {
        auto [eleSize, eleAlign] = fir::getTypeSizeAndAlignmentOrCrash(
            v.getLoc(), elementType, dl, kindMap);
        typeSize = llvm::alignTo(eleSize, eleAlign);
      }
      if (typeSize)
        return {rank, typeSize};
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Unsupported rank/type: " << v << '\n');
  return {0, 0};
}

/// If a value comes from a fir.declare of fir.pack_array,
/// follow it to the original source, otherwise return the value.
static mlir::Value unwrapPassThroughOps(mlir::Value val) {
  // Instead of unwrapping fir.declare, we may try to start
  // the analysis in this pass from fir.declare's instead
  // of the function entry block arguments. This way the loop
  // versioning would work even after FIR inlining.
  while (true) {
    if (fir::DeclareOp declare = val.getDefiningOp<fir::DeclareOp>()) {
      val = declare.getMemref();
      continue;
    }
    // fir.pack_array might be met before fir.declare - this is how
    // it is orifinally generated.
    // It might also be met after fir.declare - after the optimization
    // passes that sink fir.pack_array closer to the uses.
    if (auto packArray = val.getDefiningOp<fir::PackArrayOp>()) {
      val = packArray.getArray();
      continue;
    }
    break;
  }
  return val;
}

/// if a value comes from a fir.rebox, follow the rebox to the original source,
/// of the value, otherwise return the value
static mlir::Value unwrapReboxOp(mlir::Value val) {
  while (fir::ReboxOp rebox = val.getDefiningOp<fir::ReboxOp>()) {
    if (!fir::reboxPreservesContinuity(rebox,
                                       /*mayHaveNonDefaultLowerBounds=*/true,
                                       /*checkWhole=*/false)) {
      LLVM_DEBUG(llvm::dbgs() << "REBOX may produce non-contiguous array: "
                              << rebox << '\n');
      break;
    }
    val = rebox.getBox();
  }
  return val;
}

/// normalize a value (removing fir.declare and fir.rebox) so that we can
/// more conveniently spot values which came from function arguments
static mlir::Value normaliseVal(mlir::Value val) {
  return unwrapPassThroughOps(unwrapReboxOp(val));
}

/// Collect slice-specific state for one indexing operation during the existing
/// loop walk. A true result means that an enabled sliced access was handled
/// completely and must not enter the slice-free collection path. When slice
/// support is disabled, the function preserves the existing rejection and
/// lets that path perform its normal cleanup. A slice-free access may reject
/// an earlier sliced use of the same descriptor but otherwise continues there.
static bool collectSliceUse(fir::DoLoopOp loop, mlir::Operation *op,
                            ArgInfo &info, bool isOriginalArgument,
                            ArgsUsageInLoop &argsInLoop,
                            std::unique_ptr<LoopSliceUses> &loopUses,
                            std::optional<SliceDiscovery> &slices,
                            mlir::DominanceInfo &domInfo,
                            const fir::KindMapping &kindMap,
                            const mlir::DataLayout &dataLayout) {
  auto arrayCoor = mlir::dyn_cast<fir::ArrayCoorOp>(op);
  if (!arrayCoor || !arrayCoor.getSlice()) {
    if (slices) {
      // TODO: Support descriptors used by both sliced and slice-free accesses.
      // Until then, this combination is intentionally unsupported.
      // One slice-free direct owner makes the descriptor ineligible without
      // retaining that owner's operations or ArgInfo.
      slices->rejected.insert(info.arg);
      if (loopUses)
        if (auto found = loopUses->indices.find(info.arg);
            found != loopUses->indices.end())
          loopUses->uses[found->second].rejected = true;
    }
    return false;
  }

  if (!slices) {
    argsInLoop.cannotTransform.insert(info.arg);
    return false;
  }

  // A descriptor-wide decision cannot recover after any direct owner is
  // rejected. Keep propagating the rejection without allocating access plans
  // that preflight can never publish.
  if (slices->rejected.contains(info.arg)) {
    argsInLoop.cannotTransform.insert(info.arg);
    argsInLoop.usageInfo.erase(info.arg);
    return true;
  }
  if (!loopUses)
    loopUses = std::make_unique<LoopSliceUses>();
  auto recorded = recordSliceUse(loop, info.arg, arrayCoor, info, *loopUses,
                                 slices->nextUseOrder);
  if (!recorded) {
    argsInLoop.cannotTransform.insert(info.arg);
    argsInLoop.usageInfo.erase(info.arg);
    return true;
  }
  auto [useIndex, node, firstUse] = *recorded;

  // Dominance is owner-local, while rank and element size are invariant for
  // the concrete descriptor. Reuse the initial argument facts or a retained
  // owner's facts instead of repeating the layout query in every owner.
  if (firstUse) {
    if (!domInfo.dominates(info.arg, loop)) {
      loopUses->uses[useIndex].rejected = true;
    } else if (auto found = slices->descriptors.find(info.arg);
               found != slices->descriptors.end() &&
               !found->second.sliced.empty()) {
      const ArgInfo &previous = found->second.sliced.front()->info;
      info.rank = previous.rank;
      info.size = previous.size;
    } else if (!isOriginalArgument) {
      std::tie(info.rank, info.size) =
          getRankAndElementSize(kindMap, dataLayout, info.arg);
    }
    node->info = info;
    if (info.rank == 0 || info.size == 0)
      loopUses->uses[useIndex].rejected = true;
  }

  // Preserve the existing collection rejection until descriptor-wide
  // preflight publishes the complete frozen plan.
  argsInLoop.cannotTransform.insert(info.arg);
  argsInLoop.usageInfo.erase(info.arg);
  return true;
}

/// Return whether direct byte addressing would bypass descriptor semantics.
static bool hasUnsupportedSliceSemantics(mlir::Value value,
                                         mlir::func::FuncOp func) {
  if (fir::isa_volatile_type(value.getType()))
    return true;
  mlir::Value root = value;
  while (fir::ReboxOp rebox = root.getDefiningOp<fir::ReboxOp>()) {
    if (!fir::reboxPreservesContinuity(rebox,
                                       /*mayHaveNonDefaultLowerBounds=*/true,
                                       /*checkWhole=*/false))
      break;
    if (rebox.getOptional() || fir::isa_volatile_type(rebox.getType()) ||
        fir::isa_volatile_type(rebox.getBox().getType()))
      return true;
    root = rebox.getBox();
  }
  while (true) {
    if (fir::DeclareOp declare = root.getDefiningOp<fir::DeclareOp>()) {
      auto variable =
          mlir::cast<fir::FortranVariableOpInterface>(declare.getOperation());
      auto attrs = declare.getFortranAttrs();
      if (variable.isOptional() || fir::isa_volatile_type(declare.getType()) ||
          fir::isa_volatile_type(declare.getMemref().getType()) ||
          (attrs &&
           fir::bitEnumContainsAny(
               *attrs, fir::FortranVariableFlagsEnum::fortran_volatile)))
        return true;
      root = declare.getMemref();
      continue;
    }
    if (auto pack = root.getDefiningOp<fir::PackArrayOp>()) {
      if (fir::isa_volatile_type(pack.getType()) ||
          fir::isa_volatile_type(pack.getArray().getType()))
        return true;
      root = pack.getArray();
      continue;
    }
    break;
  }
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(root);
      blockArg && blockArg.getOwner() == &func.getBody().front()) {
    unsigned number = blockArg.getArgNumber();
    return func.getArgAttr(number, fir::getOptionalAttrName()) ||
           func.getArgAttr(number, fir::getVolatileAttrName());
  }
  return false;
}

/// Return whether a value is produced by fir.undefined.
static bool isUndefined(mlir::Value value) {
  return value && mlir::isa_and_nonnull<fir::UndefOp>(value.getDefiningOp());
}

/// Classify a slice triple exactly as generic XArrayCoor lowering does.
static SliceTripleKind classifySliceTriple(mlir::Value lower, mlir::Value upper,
                                           mlir::Value step) {
  if (isUndefined(upper))
    return SliceTripleKind::Scalar;
  if (!isUndefined(lower) && !isUndefined(step))
    return SliceTripleKind::Section;
  return SliceTripleKind::Unsupported;
}

/// Derive the address contract of one top-level FIR module.
/// A nested builtin module can be lowered either under its own contract or by
/// an ancestor module pass, so this initial slice path rejects it fail closed.
static std::optional<SliceTargetInfo>
getSliceTargetInfo(mlir::ModuleOp module,
                   const fir::KindMapping &moduleKindMap) {
  if (module->getParentOfType<mlir::ModuleOp>())
    return std::nullopt;

  auto getIndexWidth = [](mlir::ModuleOp owner) -> std::optional<unsigned> {
    llvm::StringRef layoutString;
    if (auto layout = owner->getAttrOfType<mlir::StringAttr>(
            mlir::LLVM::LLVMDialect::getDataLayoutAttrName()))
      layoutString = layout.getValue();
    auto parsedLayout = llvm::DataLayout::parse(layoutString);
    if (!parsedLayout) {
      llvm::consumeError(parsedLayout.takeError());
      return std::nullopt;
    }
    // FIR-to-LLVM lowers abstract MLIR index values to i32 only for 32-bit
    // pointers and to i64 otherwise. Generic XArrayCoor computes boxed byte
    // offsets in i64, but a 32-bit GEP observes the same low address bits.
    return parsedLayout->getPointerSizeInBits(0) == 32 ? 32u : 64u;
  };

  std::optional<unsigned> indexWidth = getIndexWidth(module);
  if (!indexWidth)
    return std::nullopt;
  return SliceTargetInfo{*indexWidth, &moduleKindMap};
}

/// Return the effective width of an integer-like slice operand. Admission
/// checks and constant-chain simulation share this cached kind-mapped result.
static unsigned getSliceOperandWidth(mlir::Type type,
                                     const SliceTargetInfo &target,
                                     SliceWidthCache &cache) {
  if (auto found = cache.find(type); found != cache.end())
    return found->second;

  unsigned width = 0;
  if (mlir::isa<mlir::IndexType>(type)) {
    width = target.indexWidth;
  } else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    width = integer.getWidth();
  } else if (auto integer = mlir::dyn_cast<fir::IntegerType>(type)) {
    assert(target.kindMapping && "slice target must retain its kind mapping");
    width = target.kindMapping->getIntegerBitsize(integer.getFKind());
  }
  cache.try_emplace(type, width);
  return width;
}

/// Return whether fir.convert can preserve this operand in the index domain.
/// The caller supplies the already computed operand width so later
/// classification can reuse it without another kind-mapping lookup.
static bool canConvertSliceOperand(mlir::Value value, unsigned width,
                                   unsigned indexWidth) {
  mlir::Type type = value.getType();
  // Generic XArrayCoor lowering sign-extends narrow integer adaptors, while
  // fir.convert preserves builtin unsigned extension. An exact target-width
  // unsigned value requires no extension, so both paths consume the same bits.
  // Wider values remain excluded by the lossless width ceiling below.
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
      integer && integer.isUnsigned() && width != indexWidth)
    return false;
  return width > 1 && width <= indexWidth;
}

/// Return whether integer widening from this source uses zero extension.
static bool isZeroExtendedSliceInteger(mlir::Type type) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && (integer.isUnsigned() ||
                     (integer.isSignless() && integer.getWidth() == 1));
}

/// Evaluate one constant integer conversion chain for the module contract.
/// Truncation, signed extension, builtin i1 extension, and FIR kind widths
/// mirror ConvertOpConversion. Every intermediate result is retained so a
/// later query can resume at the nearest previously evaluated predecessor.
static std::optional<StaticIntegerState>
evaluateStaticInteger(mlir::Value value, const SliceTargetInfo &target,
                      StaticIntegerCache &cache, SliceWidthCache &widthCache) {
  if (auto found = cache.find(value); found != cache.end())
    return found->second;

  llvm::SmallVector<fir::ConvertOp, 4> conversions;
  mlir::Value source = value;
  while (!cache.contains(source)) {
    auto convert = source.getDefiningOp<fir::ConvertOp>();
    if (!convert)
      break;
    if (!fir::isa_integer(source.getType()) ||
        !fir::isa_integer(convert.getValue().getType())) {
      cache.try_emplace(source, std::nullopt);
      break;
    }
    conversions.push_back(convert);
    source = convert.getValue();
  }

  constexpr unsigned addressIndexWidth = 64;
  std::optional<StaticIntegerState> state;
  if (auto found = cache.find(source); found != cache.end()) {
    state = found->second;
  } else {
    std::optional<llvm::APInt> constant = fir::getIntIfConstant(source);
    unsigned sourceWidth =
        getSliceOperandWidth(source.getType(), target, widthCache);
    // Canonicalization may combine a conversion chain rooted at an i1 constant
    // and materialize the direct conversion to index through signed getInt(),
    // turning the set bit into -1. Reject the root so LoopVersioning has the
    // same result whether canonicalization runs before or after this pass.
    if (constant && sourceWidth > 1) {
      unsigned retainedWidth = std::min(sourceWidth, addressIndexWidth);
      llvm::APInt retained = isZeroExtendedSliceInteger(source.getType())
                                 ? constant->zextOrTrunc(retainedWidth)
                                 : constant->sextOrTrunc(retainedWidth);
      state = StaticIntegerState{sourceWidth, std::move(retained)};
    }
    cache.try_emplace(source, state);
  }

  for (fir::ConvertOp convert : llvm::reverse(conversions)) {
    if (state) {
      mlir::Type fromType = convert.getValue().getType();
      mlir::Type toType = convert.getType();
      unsigned fromWidth = getSliceOperandWidth(fromType, target, widthCache);
      unsigned toWidth = getSliceOperandWidth(toType, target, widthCache);
      unsigned retainedFromWidth = std::min(fromWidth, addressIndexWidth);
      if (!fromWidth || !toWidth || state->width != fromWidth ||
          state->bits.getBitWidth() != retainedFromWidth) {
        state.reset();
      } else {
        unsigned retainedToWidth = std::min(toWidth, addressIndexWidth);
        llvm::APInt retained = state->bits;
        if (retainedToWidth < retainedFromWidth)
          retained = retained.trunc(retainedToWidth);
        else if (retainedToWidth > retainedFromWidth)
          retained = isZeroExtendedSliceInteger(fromType)
                         ? retained.zext(retainedToWidth)
                         : retained.sext(retainedToWidth);
        state = StaticIntegerState{toWidth, std::move(retained)};
      }
    }
    cache.try_emplace(convert.getResult(), state);
  }
  return state;
}

/// Return whether an integer value becomes positive one through fir.convert
/// under the top-level module contract. Analysis runs before modification, so
/// the result for the exact SSA value remains reusable throughout preflight.
static bool isStaticOneInteger(mlir::Value value, const SliceTargetInfo &target,
                               StaticIntegerCache &cache,
                               SliceWidthCache &widthCache) {
  std::optional<StaticIntegerState> state =
      evaluateStaticInteger(value, target, cache, widthCache);
  if (!state)
    return false;
  // Generic XArrayCoor uses a 64-bit address index and integerCast applies
  // signed extension or truncation to the final step. The direct path does not
  // materialize a proven unit step, so classify the value in that same domain.
  constexpr unsigned addressIndexWidth = 64;
  return state->bits.sextOrTrunc(addressIndexWidth).isOne();
}

/// Return whether a slice step uses a supported static-one form.
/// A chain rooted at i1 is rejected so its classification is independent of
/// canonicalization. An intermediate i1 produced from a wider integer is
/// modeled with fir.convert's zero extension, while a final i1 remains
/// unsupported because generic XArrayCoor lowering sign-extends it. Wider step
/// types need no target-index admission check because a proven unit step is not
/// materialized by the direct path.
static bool isStaticOneSliceStep(mlir::Value value,
                                 const SliceTargetInfo &target,
                                 StaticIntegerCache &cache,
                                 SliceWidthCache &widthCache) {
  return getSliceOperandWidth(value.getType(), target, widthCache) > 1 &&
         isStaticOneInteger(value, target, cache, widthCache);
}

/// Preflight descriptor properties shared by every one of its sliced accesses.
/// The returned sequence type is reused by access-level preflight so rank,
/// element size, and descriptor type are checked once per descriptor.
static mlir::FailureOr<fir::SequenceType>
analyzeSliceDescriptor(const ArgInfo &arg, unsigned indexWidth) {
  auto reject =
      [&](llvm::StringRef reason) -> mlir::FailureOr<fir::SequenceType> {
    LLVM_DEBUG(llvm::dbgs()
               << "Sliced array_coor rejected: " << reason << '\n');
    return mlir::failure();
  };

  assert((indexWidth == 32 || indexWidth == 64) &&
         "slice target must provide a supported index width");
  uint64_t maxElementSize =
      indexWidth == 32
          ? static_cast<uint64_t>(std::numeric_limits<std::int32_t>::max())
          : static_cast<uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (arg.rank > CFI_MAX_RANK || arg.size > maxElementSize)
    return reject("UnsupportedRankOrElementSize");

  auto boxType = mlir::dyn_cast<fir::BaseBoxType>(arg.arg.getType());
  if (!boxType)
    return reject("UnsupportedDescriptor");
  // This initial slice path accepts only descriptors whose direct element is
  // a sequence. Descriptors with heap or pointer storage wrappers remain on
  // the generic path.
  auto sequenceType = mlir::dyn_cast<fir::SequenceType>(boxType.getEleTy());
  if (!sequenceType || sequenceType.getDimension() != arg.rank)
    return reject("UnsupportedDescriptorElement");
  return sequenceType;
}

/// Preflight properties that belong to one physical fir.array_coor access.
/// The returned slice handle lets descriptor-wide preflight reuse slice-level
/// classification without resolving the carrier a second time.
static mlir::FailureOr<fir::SliceOp> validateSliceAccess(
    fir::ArrayCoorOp op, const ArgInfo &arg, fir::SequenceType sequenceType,
    const SliceTargetInfo &target, SliceWidthCache &widthCache) {
  auto reject = [&](llvm::StringRef reason) -> mlir::FailureOr<fir::SliceOp> {
    LLVM_DEBUG(llvm::dbgs()
               << "Sliced array_coor rejected: " << reason << '\n');
    return mlir::failure();
  };

  fir::SliceOp slice = op.getSlice().getDefiningOp<fir::SliceOp>();
  if (!slice)
    return reject("UnsupportedSliceCarrier");
  if (fir::unwrapRefType(op.getType()) != sequenceType.getEleTy())
    return reject("ResultElementTypeMismatch");
  if (op.getIndices().size() != arg.rank)
    return reject("UnsupportedIndexConvention");
  if (mlir::Value shape = op.getShape()) {
    auto shapeOp = shape.getDefiningOp<fir::ShapeOp>();
    if (!shapeOp)
      return reject("UnsupportedShapeCarrier");
  }

  for (mlir::Value index : op.getIndices()) {
    unsigned width = getSliceOperandWidth(index.getType(), target, widthCache);
    if (!canConvertSliceOperand(index, width, target.indexWidth))
      return reject("UnsupportedIndexType");
  }
  return slice;
}

/// Preflight and decode one static-unit fir.slice without modifying the IR.
/// A descriptor owner caches the result by operation so all accesses sharing
/// this slice reuse the same classification and width proofs.
static mlir::FailureOr<SliceFacts>
analyzeSlice(fir::SliceOp slice, unsigned rank, const SliceTargetInfo &target,
             StaticIntegerCache &cache, SliceWidthCache &widthCache) {
  auto reject = [&](llvm::StringRef reason) -> mlir::FailureOr<SliceFacts> {
    LLVM_DEBUG(llvm::dbgs()
               << "Sliced array_coor rejected: " << reason << '\n');
    return mlir::failure();
  };

  // TODO: Support the remaining valid fir.slice forms. This initial slice
  // implementation intentionally leaves component paths, a leading scalar
  // dimension, and non-unit section steps on the generic path.
  if (!slice.getFields().empty())
    return reject("UnsupportedComponentPath");
  SliceFacts facts;
  facts.reserve(rank);
  mlir::ValueRange triples = slice.getTriples();
  for (unsigned dim = 0; dim < rank; ++dim) {
    mlir::Value lower = triples[3 * dim];
    mlir::Value upper = triples[3 * dim + 1];
    mlir::Value step = triples[3 * dim + 2];
    SliceTripleKind kind = classifySliceTriple(lower, upper, step);
    if (kind == SliceTripleKind::Unsupported)
      return reject("UnsupportedSliceTriple");
    if (dim == 0 && kind != SliceTripleKind::Section)
      return reject("LeadingScalarDimension");

    bool lowerIsOne = false;
    if (kind == SliceTripleKind::Section) {
      // A proven source lower one contributes no adjustment and is never
      // materialized by the direct path, so its source width is irrelevant.
      lowerIsOne = isStaticOneInteger(lower, target, cache, widthCache);
      if (!lowerIsOne) {
        unsigned lowerWidth =
            getSliceOperandWidth(lower.getType(), target, widthCache);
        if (!canConvertSliceOperand(lower, lowerWidth, target.indexWidth))
          return reject("UnsupportedSectionLowerBound");
      }
      // Generic XArrayCoor sign-extends integer operands. The one-bit value
      // `1` therefore denotes -1 after widening and is not a unit step.
      if (!isStaticOneSliceStep(step, target, cache, widthCache))
        return reject("NonUnitSectionStep");
    }
    facts.push_back({kind, lowerIsOne});
  }
  return facts;
}

/// some FIR operations accept a fir.shape, a fir.shift or a fir.shapeshift.
/// fir.shift and fir.shapeshift allow us to extract lower bounds
/// if lowerbounds cannot be found, return nullptr
static mlir::Value tryGetLowerBoundsFromShapeLike(mlir::Value shapeLike,
                                                  unsigned dim) {
  mlir::Value lowerBound{nullptr};
  if (auto shift = shapeLike.getDefiningOp<fir::ShiftOp>())
    lowerBound = shift.getOrigins()[dim];
  if (auto shapeShift = shapeLike.getDefiningOp<fir::ShapeShiftOp>())
    lowerBound = shapeShift.getOrigins()[dim];
  return lowerBound;
}

/// attempt to get the array lower bounds of dimension dim of the memref
/// argument to a fir.array_coor op
/// 0 <= dim < rank
/// May return nullptr if no lower bounds can be determined
static mlir::Value getLowerBound(fir::ArrayCoorOp coop, unsigned dim) {
  // 1) try to get from the shape argument to fir.array_coor
  if (mlir::Value shapeLike = coop.getShape())
    if (mlir::Value lb = tryGetLowerBoundsFromShapeLike(shapeLike, dim))
      return lb;

  // It is important not to try to read the lower bound from the box, because
  // in the FIR lowering, boxes will sometimes contain incorrect lower bound
  // information

  // out of ideas
  return {};
}

/// gets the i'th index from array coordinate operation op
/// dim should range between 0 and rank - 1
static mlir::Value getIndex(fir::FirOpBuilder &builder, mlir::Operation *op,
                            unsigned dim) {
  if (fir::CoordinateOp coop = mlir::dyn_cast<fir::CoordinateOp>(op))
    return coop.getCoor()[dim];

  fir::ArrayCoorOp coop = mlir::dyn_cast<fir::ArrayCoorOp>(op);
  assert(coop &&
         "operation must be either fir.coordiante_of or fir.array_coor");

  // fir.coordinate_of indices start at 0: adjust these indices to match by
  // subtracting the lower bound
  mlir::Value index = coop.getIndices()[dim];
  mlir::Value lb = getLowerBound(coop, dim);
  if (!lb)
    // assume a default lower bound of one
    lb = builder.createIntegerConstant(coop.getLoc(), index.getType(), 1);

  // index_0 = index - lb;
  if (lb.getType() != index.getType())
    lb = builder.createConvert(coop.getLoc(), index.getType(), lb);
  return mlir::arith::SubIOp::create(builder, coop.getLoc(), index, lb);
}

/// Convert one cloned slice operand to the index type before arithmetic.
/// Preflight proved that this conversion preserves the effective index-domain
/// bit pattern used by generic lowering.
static mlir::Value materializeIndex(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value value) {
  mlir::Type indexType = builder.getIndexType();
  if (value.getType() == indexType)
    return value;
  return builder.createConvert(loc, indexType, value);
}

/// Extract a byte-address base for one accepted sliced descriptor.
/// Descriptor order is controlled by the caller, and access-local arithmetic
/// remains at each cloned indexing operation.
static mlir::Value createSliceByteBase(fir::FirOpBuilder &builder,
                                       mlir::Location loc, const ArgInfo &arg) {
  fir::SequenceType::Shape shape{fir::SequenceType::getUnknownExtent()};
  mlir::Type byteType = builder.getIntegerType(8);
  mlir::Type byteArrayType = fir::SequenceType::get(shape, byteType);
  mlir::Type byteBoxType = fir::BoxType::get(byteArrayType);
  mlir::Value byteBox = builder.createConvert(loc, byteBoxType, arg.arg);
  return fir::BoxAddrOp::create(builder, loc, builder.getRefType(byteArrayType),
                                byteBox);
}

/// Recreate one accepted sliced access in the descriptor byte domain.
/// Every input-dependent check completed before modification, so this function
/// only resolves clone-local operands and emits the frozen unit-step formula.
static mlir::Value
rewriteSliceAccess(fir::FirOpBuilder &builder, fir::ArrayCoorOp access,
                   fir::SliceOp slice, const SliceFacts &facts,
                   const ArgInfo &arg, mlir::Value byteBase, mlir::Value one) {
  assert(facts.size() == arg.rank &&
         "slice facts must describe every source dimension");
  assert(arg.rank && one && one.getType() == builder.getIndexType() &&
         "slice offset arithmetic requires a dominating index one");
  builder.setInsertionPoint(access);
  mlir::Location loc = access.getLoc();
  mlir::Value totalBytes;
  assert(slice && "an accepted access must retain its slice");
  mlir::ValueRange triples = slice.getTriples();
  assert(triples.size() == 3 * facts.size() &&
         "slice triples must match the frozen source rank");

  for (unsigned dim = 0; dim < facts.size(); ++dim) {
    const SliceDimensionFacts &dimFacts = facts[dim];
    mlir::Value index =
        materializeIndex(builder, loc, access.getIndices()[dim]);
    mlir::Value offset =
        builder.createOrFold<mlir::arith::SubIOp>(loc, index, one);

    // Generic XArrayCoor lowering adds (sliceLower - sourceLower) for a
    // retained section. This patch accepts only implicit source lower one and
    // statically unit section steps.
    if (dimFacts.kind == SliceTripleKind::Section && !dimFacts.lowerIsOne) {
      mlir::Value sliceLower = materializeIndex(builder, loc, triples[3 * dim]);
      mlir::Value adjustment =
          builder.createOrFold<mlir::arith::SubIOp>(loc, sliceLower, one);
      offset =
          builder.createOrFold<mlir::arith::AddIOp>(loc, offset, adjustment);
    }

    // The fast-edge predicate proves the dimension-zero byte stride equals
    // the element size. Outer dimensions retain descriptor byte strides.
    mlir::Value byteStride =
        dim == 0 ? arg.elemSize : arg.dims[dim]->getResult(2);
    // createOrFold removes a statically zero contribution immediately.
    mlir::Value contribution =
        builder.createOrFold<mlir::arith::MulIOp>(loc, byteStride, offset);
    totalBytes = totalBytes ? builder.createOrFold<mlir::arith::AddIOp>(
                                  loc, totalBytes, contribution)
                            : contribution;
  }

  assert(byteBase && "a sliced access requires a byte-address base");
  assert(totalBytes && "a rewritten slice must produce a byte offset");
  mlir::Type byteType = builder.getIntegerType(8);
  mlir::Value byteAddress =
      fir::CoordinateOp::create(builder, loc, builder.getRefType(byteType),
                                byteBase, mlir::ValueRange{totalBytes});
  return builder.createConvert(loc, access.getType(), byteAddress);
}

/// Run complete descriptor-wide slice preflight before any IR modification.
/// The function rejects nested or incomplete owner sets, preflights every
/// physical sliced access, and freezes accepted plans for final owner selection
/// after the existing slice-free cleanup has completed.
/// Collection cannot accept uses incrementally because a later slice-free or
/// unsupported access, or a nested owner, can invalidate every previously
/// collected use of the descriptor. Running preflight after discovery preserves
/// descriptor-wide atomicity without modifying and rolling back IR.
static void freezeSliceOwnership(
    llvm::ArrayRef<fir::DoLoopOp> originalLoops,
    llvm::DenseMap<mlir::Value, DescriptorUses> &descriptorUses,
    const llvm::SmallDenseSet<mlir::Value, 4> &rejectedDescriptors,
    llvm::SmallVectorImpl<std::unique_ptr<UseNode>> &sliceNodes,
    mlir::func::FuncOp func, mlir::ModuleOp module,
    const fir::KindMapping &moduleKindMap) {
  assert(!sliceNodes.empty() &&
         "slice ownership requires at least one sliced use");

  std::optional<SliceTargetInfo> target =
      getSliceTargetInfo(module, moduleKindMap);
  if (!target) {
    LLVM_DEBUG(llvm::dbgs()
               << "Sliced array_coor rejected: UnsupportedTargetLayout\n");
    descriptorUses.clear();
    sliceNodes.clear();
    return;
  }

  llvm::SmallVector<DescriptorUses *, 4> multiOwnerDescriptors;
  for (auto &[descriptor, uses] : descriptorUses) {
    if (uses.sliced.size() > 1 && !rejectedDescriptors.contains(descriptor))
      multiOwnerDescriptors.push_back(&uses);
  }

  if (!multiOwnerDescriptors.empty()) {
    // Loop post-order defines compact subtree intervals. Build them only when
    // a complete sliced descriptor has multiple owners and can therefore have
    // an ancestor-descendant ownership conflict.
    /// Inclusive range occupied by one loop subtree in loop post-order.
    struct LoopInterval {
      /// First post-order position belonging to the subtree.
      size_t begin;
      /// Post-order position of the loop itself.
      size_t end;
    };
    llvm::DenseMap<fir::DoLoopOp, LoopInterval> intervals;
    llvm::DenseMap<fir::DoLoopOp, size_t> subtreeBegins;
    for (auto [index, loop] : llvm::enumerate(originalLoops)) {
      size_t end = index;
      size_t begin = end;
      if (auto nested = subtreeBegins.find(loop); nested != subtreeBegins.end())
        begin = nested->second;
      intervals.try_emplace(loop, LoopInterval{begin, end});
      if (fir::DoLoopOp parent = loop->getParentOfType<fir::DoLoopOp>()) {
        auto [parentBegin, inserted] = subtreeBegins.try_emplace(parent, begin);
        if (!inserted)
          parentBegin->second = std::min(parentBegin->second, begin);
      }
    }

    for (DescriptorUses *uses : multiOwnerDescriptors) {
      std::optional<size_t> previousEnd;
      for (UseNode *use : uses->sliced) {
        fir::DoLoopOp owner = use->loop;
        auto ownerInterval = intervals.find(owner);
        if (ownerInterval == intervals.end()) {
          uses->nested = true;
          break;
        }
        LoopInterval interval = ownerInterval->second;
        if (previousEnd && interval.begin <= *previousEnd) {
          uses->nested = true;
          break;
        }
        previousEnd = interval.end;
      }
    }
  }

  // Freeze only descriptors whose complete set of independent direct owners
  // is made of supported slices. A descriptor is marked accepted only after
  // every physical access of every owner has passed.
  StaticIntegerCache staticIntegerCache;
  SliceWidthCache widthCache;
  // A direct fir.slice may dominate several independent owners. Cache both
  // successful and rejected classifications so its immutable triples are
  // decoded once for the whole function.
  llvm::DenseMap<mlir::Operation *, std::optional<SliceFacts>> sliceFactsCache;
  for (const std::unique_ptr<UseNode> &firstNode : sliceNodes) {
    mlir::Value descriptor = firstNode->info.arg;
    auto descriptorIt = descriptorUses.find(descriptor);
    if (descriptorIt == descriptorUses.end() ||
        descriptorIt->second.sliced.empty())
      continue;
    const DescriptorUses &descriptorInfo = descriptorIt->second;
    if (firstNode.get() != descriptorInfo.sliced.front())
      continue;
    mlir::ArrayRef<UseNode *> uses = descriptorInfo.sliced;

    // A slice-free or rejected owner makes the descriptor incomplete. A
    // nested owner would require remapping facts after cloning the descendant.
    // Both cases prevent publishing plans for the whole descriptor.
    bool rejected =
        descriptorInfo.nested || rejectedDescriptors.contains(descriptor);
    if (rejected || hasUnsupportedSliceSemantics(descriptor, func))
      continue;

    // Run descriptor-level preflight once and reuse its verified sequence type
    // for every physical access of every owner.
    mlir::FailureOr<fir::SequenceType> sequenceType =
        analyzeSliceDescriptor(firstNode->info, target->indexWidth);
    if (mlir::failed(sequenceType))
      continue;

    // Preflight every owner and access before marking any owner accepted, so a
    // late failure rejects the complete descriptor without partial plans.
    for (UseNode *use : uses) {
      // Store facts for each distinct fir.slice once per owner. Access plans
      // retain stable indices rather than pointers into the growable vector.
      llvm::DenseMap<mlir::Operation *, unsigned> factIndices;
      for (SliceAccessPlan &plan : use->accesses) {
        fir::ArrayCoorOp arrayCoor = plan.source;
        assert(arrayCoor && arrayCoor.getSlice() &&
               "collected slice access must remain attached");
        // Replacing this operation would bypass its volatile access semantics.
        if (fir::isa_volatile_type(arrayCoor.getType())) {
          rejected = true;
          break;
        }
        // Access-level preflight verifies the carrier, shape, indices, and
        // result type and returns the concrete fir.slice for shared analysis.
        mlir::FailureOr<fir::SliceOp> slice = validateSliceAccess(
            arrayCoor, use->info, *sequenceType, *target, widthCache);
        if (mlir::failed(slice)) {
          rejected = true;
          break;
        }
        auto [fact, firstUse] = factIndices.try_emplace(slice->getOperation(),
                                                        use->sliceFacts.size());
        if (firstUse) {
          // Reuse slice-level preflight across independent owners. The cache
          // retains failures too, so an unsupported slice is never reanalyzed.
          auto cached = sliceFactsCache.find(slice->getOperation());
          if (cached == sliceFactsCache.end()) {
            mlir::FailureOr<SliceFacts> facts =
                analyzeSlice(*slice, use->info.rank, *target,
                             staticIntegerCache, widthCache);
            cached = sliceFactsCache
                         .try_emplace(
                             slice->getOperation(),
                             mlir::failed(facts)
                                 ? std::nullopt
                                 : std::optional<SliceFacts>(std::move(*facts)))
                         .first;
          }
          if (!cached->second) {
            rejected = true;
            break;
          }
          // Copy immutable facts into owner-local storage whose lifetime covers
          // later clone rewriting.
          use->sliceFacts.push_back(*cached->second);
        }
        // Freeze the exact relation between this access and its owner-local
        // facts without retaining a pointer that vector growth could
        // invalidate.
        plan.factsIndex = fact->second;
      }
      if (rejected)
        break;
    }

    // Any owner or access failure rejects the complete descriptor plan.
    if (rejected)
      continue;

    // Commit ownership atomically only after every access has complete facts.
    for (UseNode *use : uses) {
      assert(!use->accesses.empty() &&
             llvm::all_of(use->accesses,
                          [](const SliceAccessPlan &plan) {
                            return plan.factsIndex.has_value();
                          }) &&
             "accepted use must freeze every direct access");
      use->info.sliceNode = use;
    }
  }

  // Only accepted descriptor groups and nodes are needed after preflight.
  // Moving unique_ptr values does not move the accepted pointees referenced by
  // DescriptorUses.
  for (auto &entry : descriptorUses) {
    DescriptorUses &descriptorInfo = entry.second;
    if (!descriptorInfo.sliced.empty() &&
        !descriptorInfo.sliced.front()->info.sliceNode)
      descriptorInfo.sliced.clear();
  }
  llvm::erase_if(sliceNodes, [](const std::unique_ptr<UseNode> &node) {
    return !node->info.sliceNode;
  });
}

/// Publish frozen slice plans after slice-free ownership is final.
/// Publishing means adding fully preflighted plans to argsInLoops so the
/// common loop-selection and rewriting phases can consume them.
/// A descriptor is published for all of its independent owners only when none
/// of those owners retains a slice-free rewrite. Iterating the frozen groups
/// avoids another IR traversal and preserves descriptor-wide atomicity.
static void publishSliceOwnership(
    LoopUsageMap &argsInLoops,
    llvm::DenseMap<mlir::Value, DescriptorUses> &descriptorUses,
    llvm::SmallVectorImpl<std::unique_ptr<UseNode>> &sliceNodes) {
  /// Complete owner records retained for one descriptor publication.
  /// Pointers remain stable because publication modifies only the nested
  /// usageInfo maps and never adds entries to argsInLoops.
  struct Publication {
    mlir::Value descriptor;
    llvm::SmallVector<std::pair<UseNode *, ArgsUsageInLoop *>, 2> owners;
  };

  // Decide every group against the same pre-publication usageInfo snapshot.
  // Retain the owner records found during that decision so publication neither
  // repeats lookups nor admits a strict subset if a group is incomplete.
  llvm::SmallVector<Publication, 4> publications;
  for (auto &[descriptor, descriptorInfo] : descriptorUses) {
    if (descriptorInfo.sliced.empty())
      continue;
    assert(llvm::all_of(
               descriptorInfo.sliced,
               [](const UseNode *use) { return use->info.sliceNode == use; }) &&
           "published descriptor must retain complete frozen owners");

    Publication publication{descriptor, {}};
    publication.owners.reserve(descriptorInfo.sliced.size());
    for (UseNode *use : descriptorInfo.sliced) {
      auto loop = argsInLoops.find(use->loop);
      // TODO: Support owners that combine sliced and slice-free descriptors.
      // Until then, a missing owner record or any retained slice-free rewrite
      // prevents publication of the complete sliced descriptor group.
      if (loop == argsInLoops.end() || !loop->second.usageInfo.empty()) {
        publication.owners.clear();
        break;
      }
      publication.owners.emplace_back(use, &loop->second);
    }
    if (publication.owners.size() == descriptorInfo.sliced.size())
      publications.push_back(std::move(publication));
  }

  // Publish only after every descriptor has observed the unchanged final
  // slice-free state. Multiple sliced descriptors may then share one owner.
  llvm::SmallDenseSet<UseNode *, 4> publishedNodes;
  for (Publication &publication : publications) {
    for (auto [use, owner] : publication.owners) {
      auto [entry, inserted] =
          owner->usageInfo.try_emplace(publication.descriptor, use->info);
      (void)entry;
      assert(inserted && "a frozen slice must have no slice-free duplicate");
      (void)inserted;
      // Rejection propagation is complete. Keep the final owner summary
      // consistent with the successfully published descriptor.
      owner->cannotTransform.remove(publication.descriptor);
      publishedNodes.insert(use);
    }
  }

  descriptorUses.clear();
  llvm::erase_if(sliceNodes, [&](const std::unique_ptr<UseNode> &node) {
    return !publishedNodes.contains(node.get());
  });
}

#ifndef NDEBUG
/// Assert that every frozen plan still names its original sliced use.
/// This debug-only check catches accidental analysis/emission drift without
/// adding release work or changing descriptor-local fail-closed decisions.
static void validateFrozenAccesses(const UseNode &use) {
  assert(use.loop && use.info.arg && !use.accesses.empty() &&
         "accepted slice use must have an owner, descriptor, and accesses");
  for (const SliceAccessPlan &plan : use.accesses) {
    fir::ArrayCoorOp access = plan.source;
    assert(access && access->getBlock() && access.getSlice() &&
           "a planned sliced access must remain attached");
    assert(access->getParentOfType<fir::DoLoopOp>() == use.loop &&
           "a planned access must retain its immediate owner");
    assert(access.getMemref() == use.info.arg &&
           "a planned access must retain its descriptor");
    assert(plan.factsIndex && *plan.factsIndex < use.sliceFacts.size() &&
           use.sliceFacts[*plan.factsIndex].size() == use.info.rank &&
           "a planned access must retain complete immutable facts");
  }
}
#endif

/// Rewrite every frozen sliced access in one cloned owner.
/// Descriptor bases and physical accesses are consumed in their frozen order;
/// one shared index constant dominates all access-local byte arithmetic.
static bool rewriteSliceOwner(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::IndexType indexType,
                              llvm::MutableArrayRef<ArgInfo> args,
                              mlir::IRMapping &cloneMap) {
  assert(
      llvm::all_of(
          args, [](const ArgInfo &arg) { return arg.sliceNode != nullptr; }) &&
      "a sliced owner must contain only sliced descriptors");

  mlir::Value one = builder.createIntegerConstant(loc, indexType, 1);
  bool changed = false;
  for (ArgInfo &arg : args) {
    // Only stride zero is needed by the branch predicate. Materialize the
    // remaining descriptor metadata in the fast branch where it is consumed,
    // so the fallback path does not perform unnecessary descriptor reads.
    for (unsigned dim = 1; dim < arg.rank; ++dim) {
      mlir::Value dimIndex = builder.createIntegerConstant(loc, indexType, dim);
      arg.dims[dim] = fir::BoxDimsOp::create(builder, loc, indexType, indexType,
                                             indexType, arg.arg, dimIndex);
    }
    mlir::Value byteBase = createSliceByteBase(builder, loc, arg);
    auto insertionPoint = builder.saveInsertionPoint();
    for (SliceAccessPlan &plan : arg.sliceNode->accesses) {
      assert(plan.factsIndex &&
             *plan.factsIndex < arg.sliceNode->sliceFacts.size() &&
             "mapped access must retain its frozen facts");
      mlir::Operation *mapped = cloneMap.lookup(plan.source.getOperation());
      auto arrayCoor = mlir::cast<fir::ArrayCoorOp>(mapped);
      assert(arrayCoor.getMemref() == arg.arg && arrayCoor.getSlice() &&
             "mapped access must preserve its frozen plan");
      fir::SliceOp slice = arrayCoor.getSlice().getDefiningOp<fir::SliceOp>();
      assert(slice && "mapped access must retain its slice operation");
      const SliceFacts &facts = arg.sliceNode->sliceFacts[*plan.factsIndex];
      mlir::Value replacement = rewriteSliceAccess(builder, arrayCoor, slice,
                                                   facts, arg, byteBase, one);
      arrayCoor.getResult().replaceAllUsesWith(replacement);
      arrayCoor.erase();
      if (slice->use_empty())
        slice.erase();
      changed = true;
    }
    builder.restoreInsertionPoint(insertionPoint);
  }
  return changed;
}

/// Materialize stride-zero metadata for a sliced owner and return its combined
/// contiguity predicate. Metadata used only for address construction is
/// deferred to the fast branch.
static mlir::Value prepareSliceOwner(fir::FirOpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::IndexType indexType,
                                     llvm::MutableArrayRef<ArgInfo> args) {
  mlir::Value condition;
  for (ArgInfo &arg : args) {
    assert(arg.sliceNode && "a sliced owner must retain frozen access facts");
    mlir::Value dimIndex = builder.createIntegerConstant(loc, indexType, 0);
    arg.dims[0] = fir::BoxDimsOp::create(builder, loc, indexType, indexType,
                                         indexType, arg.arg, dimIndex);
    // The sliced byte formula substitutes the element size for stride zero.
    // This predicate proves that substitution for every access of descriptor.
    arg.elemSize = builder.createIntegerConstant(loc, indexType, arg.size);
    mlir::Value compare = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, arg.dims[0].getResult(2),
        arg.elemSize);
    condition = condition ? mlir::arith::AndIOp::create(builder, loc, compare,
                                                        condition)
                          : compare;
  }
  return condition;
}

void LoopVersioningPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::func::FuncOp func = getOperation();

  // First look for arguments with assumed shape = unknown extent in the lowest
  // dimension.
  LLVM_DEBUG(llvm::dbgs() << "Func-name:" << func.getSymName() << "\n");
  mlir::Block::BlockArgListType args = func.getArguments();
  mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
  fir::KindMapping kindMap = fir::getKindMapping(module);
  mlir::SmallVector<ArgInfo, 4> argsOfInterest;
  std::optional<mlir::DataLayout> dl = fir::support::getOrSetMLIRDataLayout(
      module, /*allowDefaultLayout=*/false);
  if (!dl)
    mlir::emitError(module.getLoc(),
                    "data layout attribute is required to perform " DEBUG_TYPE
                    "pass");
  for (auto &arg : args) {
    // Optional arguments must be checked for IsPresent before
    // looking for the bounds. They are unsupported for the time being.
    if (func.getArgAttrOfType<mlir::UnitAttr>(arg.getArgNumber(),
                                              fir::getOptionalAttrName())) {
      LLVM_DEBUG(llvm::dbgs() << "OPTIONAL is not supported\n");
      continue;
    }

    auto [rank, typeSize] =
        getRankAndElementSize(kindMap, *dl, arg, /*isArgument=*/true);
    if (rank != 0 && typeSize != 0)
      argsOfInterest.push_back({arg, typeSize, rank, {}, {}});
  }

  if (argsOfInterest.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No suitable arguments.\n=== End " DEBUG_TYPE " ===\n");
    return;
  }

  // A list of all loops in the function in post-order.
  mlir::SmallVector<fir::DoLoopOp> originalLoops;
  // Information about the arguments usage by the instructions
  // immediately nested in a loop.
  LoopUsageMap argsInLoops;

  // Keep slice state completely absent when the feature is disabled.
  // Accepted nodes remain stable through transformation because ArgInfo
  // stores pointers into this owner.
  std::optional<SliceDiscovery> slices;
  if (enableSlices)
    slices.emplace();
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();

  // Traverse the loops in post-order and group indexing operations by their
  // nearest enclosing do_loop.
  func.walk([&](fir::DoLoopOp loop) {
    mlir::Block &body = *loop.getBody();
    auto &argsInLoop = argsInLoops[loop];
    // Allocate per-loop lookup state only after finding a sliced access.
    std::unique_ptr<LoopSliceUses> loopSlices;
    originalLoops.push_back(loop);
    body.walk([&](mlir::Operation *op) {
      // Support either fir.array_coor or fir.coordinate_of.
      if (!mlir::isa<fir::ArrayCoorOp, fir::CoordinateOp>(op))
        return;
      // Process only operations immediately nested in the current loop.
      if (op->getParentOfType<fir::DoLoopOp>() != loop)
        return;
      mlir::Value operand = op->getOperand(0);
      for (auto a : argsOfInterest) {
        if (a.arg == normaliseVal(operand)) {
          bool isOriginalArgument = a.arg == operand;
          // Use the reboxed value, not the block arg when re-creating the
          // loop.
          a.arg = operand;

          if (collectSliceUse(loop, op, a, isOriginalArgument, argsInLoop,
                              loopSlices, slices, domInfo, kindMap, *dl))
            break;

          // Check that the operand dominates the loop?
          // If this is the case, record such operands in argsInLoop.cannot-
          // Transform, so that they disable the transformation for the parent
          /// loops as well.
          if (!domInfo.dominates(a.arg, loop))
            argsInLoop.cannotTransform.insert(a.arg);

          // We need to compute the rank and element size
          // based on the operand, not the original argument,
          // because array slicing may affect it.
          std::tie(a.rank, a.size) = getRankAndElementSize(kindMap, *dl, a.arg);
          if (a.rank == 0 || a.size == 0) {
            argsInLoop.cannotTransform.insert(a.arg);
          }

          if (argsInLoop.cannotTransform.contains(a.arg)) {
            // Remove any previously recorded usage, if any.
            argsInLoop.usageInfo.erase(a.arg);
            break;
          }

          // Record the a.arg usage, if not recorded yet.
          argsInLoop.usageInfo.try_emplace(a.arg, a);
          break;
        }
      }
    });

    // Move viable sliced nodes into function-lifetime storage. A locally
    // rejected group is represented by one descriptor value only.
    if (loopSlices)
      for (SliceUse &use : loopSlices->uses) {
        mlir::Value descriptor = use.node->info.arg;
        if (use.rejected) {
          slices->rejected.insert(descriptor);
          continue;
        }
        slices->descriptors[descriptor].sliced.push_back(use.node.get());
        slices->nodes.push_back(std::move(use.node));
      }
  });

  if (slices) {
    // Run complete slice preflight before any IR modification and freeze every
    // accepted descriptor-wide rewrite plan.
    if (!slices->nodes.empty())
      freezeSliceOwnership(originalLoops, slices->descriptors, slices->rejected,
                           slices->nodes, func, module, kindMap);
    // Rejection summaries are needed only while preflight is deciding groups.
    slices->rejected.clear();
    if (slices->nodes.empty())
      slices.reset();
  }

  // Dump loops info after initial collection.
  LLVM_DEBUG({
    llvm::dbgs() << "Initial usage info:\n";
    for (fir::DoLoopOp loop : originalLoops) {
      auto &argsInLoop = argsInLoops[loop];
      argsInLoop.dump(loop);
    }
  });

  // Clear argument usage for parent loops if an inner loop
  // contains a non-transformable usage.
  for (fir::DoLoopOp loop : originalLoops) {
    auto &argsInLoop = argsInLoops[loop];
    if (argsInLoop.cannotTransform.empty())
      continue;

    fir::DoLoopOp parent = loop;
    while ((parent = parent->getParentOfType<fir::DoLoopOp>()))
      argsInLoops[parent].eraseUsage(argsInLoop.cannotTransform);
  }

  // If an argument access can be optimized in a loop and
  // its descendant loop, then it does not make sense to
  // generate the contiguity check for the descendant loop.
  // The check will be produced as part of the ancestor
  // loop's transformation. So we can clear the argument
  // usage for all descendant loops.
  for (fir::DoLoopOp loop : originalLoops) {
    auto &argsInLoop = argsInLoops[loop];
    if (argsInLoop.usageInfo.empty())
      continue;

    loop.getBody()->walk([&](fir::DoLoopOp dloop) {
      argsInLoops[dloop].eraseUsage(argsInLoop.usageInfo);
    });
  }

  if (slices)
    publishSliceOwnership(argsInLoops, slices->descriptors, slices->nodes);

  LLVM_DEBUG({
    llvm::dbgs() << "Final usage info:\n";
    for (fir::DoLoopOp loop : originalLoops) {
      auto &argsInLoop = argsInLoops[loop];
      argsInLoop.dump(loop);
    }
  });

  // Reduce the collected information to a list of loops
  // with attached arguments usage information.
  // The list must hold the loops in post order, so that
  // the inner loops are transformed before the outer loops.
  struct OpsWithArgs {
    mlir::Operation *op;
    mlir::SmallVector<ArgInfo, 4> argsAndDims;
    /// Whether this owner contains at least one frozen slice rewrite.
    bool hasSlices = false;
  };
  mlir::SmallVector<OpsWithArgs, 4> loopsOfInterest;
  for (fir::DoLoopOp loop : originalLoops) {
    auto &argsInLoop = argsInLoops[loop];
    if (argsInLoop.usageInfo.empty())
      continue;
    OpsWithArgs info;
    info.op = loop;
    for (auto &arg : argsInLoop.usageInfo) {
      info.argsAndDims.push_back(arg.second);
      info.hasSlices |= arg.second.sliceNode != nullptr;
    }
    // Pointer-keyed ordering is preserved for slice-free owners. A sliced
    // owner uses stable first-access order for every descriptor guard and base
    // emitted by the new path.
    if (info.hasSlices)
      llvm::sort(info.argsAndDims,
                 [](const ArgInfo &left, const ArgInfo &right) {
                   return left.firstUseOrder < right.firstUseOrder;
                 });
    loopsOfInterest.emplace_back(std::move(info));
  }

  if (loopsOfInterest.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No loops to transform.\n=== End " DEBUG_TYPE " ===\n");
    return;
  }

  // If we get here, there are loops to process.
  fir::FirOpBuilder builder{module, std::move(kindMap)};
  mlir::Location loc = builder.getUnknownLoc();
  mlir::IndexType idxTy = builder.getIndexType();

  LLVM_DEBUG(llvm::dbgs() << "Func Before transformation:\n");
  LLVM_DEBUG(func->dump());

  LLVM_DEBUG(llvm::dbgs() << "loopsOfInterest: " << loopsOfInterest.size()
                          << "\n");
  for (auto op : loopsOfInterest) {
    LLVM_DEBUG(op.op->dump());
#ifndef NDEBUG
    if (op.hasSlices)
      for (const ArgInfo &arg : op.argsAndDims)
        if (arg.sliceNode)
          validateFrozenAccesses(*arg.sliceNode);
#endif
    builder.setInsertionPoint(op.op);

    mlir::Value allCompares = nullptr;
    if (op.hasSlices) {
      allCompares = prepareSliceOwner(builder, loc, idxTy, op.argsAndDims);
    } else {
      // Ensure all of the arrays are unit-stride.
      for (auto &arg : op.argsAndDims) {
        // Fetch all the dimensions of the array, except the last dimension.
        // Always fetch the first dimension, however, so set ndims = 1 if
        // we have one dim
        unsigned ndims = arg.rank;
        for (unsigned i = 0; i < ndims; i++) {
          mlir::Value dimIdx = builder.createIntegerConstant(loc, idxTy, i);
          arg.dims[i] = fir::BoxDimsOp::create(builder, loc, idxTy, idxTy,
                                               idxTy, arg.arg, dimIdx);
        }
        // We only care about lowest order dimension, here.
        mlir::Value elemSize =
            builder.createIntegerConstant(loc, idxTy, arg.size);
        mlir::Value cmp = mlir::arith::CmpIOp::create(
            builder, loc, mlir::arith::CmpIPredicate::eq,
            arg.dims[0].getResult(2), elemSize);
        if (!allCompares) {
          allCompares = cmp;
        } else {
          allCompares =
              mlir::arith::AndIOp::create(builder, loc, cmp, allCompares);
        }
      }
    }

    auto ifOp =
        fir::IfOp::create(builder, loc, op.op->getResultTypes(), allCompares,
                          /*withElse=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    LLVM_DEBUG(llvm::dbgs() << "Creating cloned loop\n");
    mlir::Operation *clonedLoop;
    bool changed = false;
    if (op.hasSlices) {
      mlir::IRMapping cloneMap;
      clonedLoop = op.op->clone(cloneMap);
      changed =
          rewriteSliceOwner(builder, loc, idxTy, op.argsAndDims, cloneMap);
    } else {
      clonedLoop = op.op->clone();
      // Keep the existing slice-free emitter structurally unchanged.
      for (auto &arg : op.argsAndDims) {
        fir::SequenceType::Shape newShape;
        newShape.push_back(fir::SequenceType::getUnknownExtent());
        auto elementType = fir::unwrapSeqOrBoxedSeqType(arg.arg.getType());
        mlir::Type arrTy = fir::SequenceType::get(newShape, elementType);
        mlir::Type boxArrTy = fir::BoxType::get(arrTy);
        mlir::Type refArrTy = builder.getRefType(arrTy);
        auto carg = fir::ConvertOp::create(builder, loc, boxArrTy, arg.arg);
        auto caddr = fir::BoxAddrOp::create(builder, loc, refArrTy, carg);
        auto insPt = builder.saveInsertionPoint();
        // Use caddr instead of arg.
        clonedLoop->walk([&](mlir::Operation *coop) {
          if (!mlir::isa<fir::CoordinateOp, fir::ArrayCoorOp>(coop))
            return;
          // Reduce the multi-dimensioned index to a single index.
          // This is required becase fir arrays do not support multiple
          // dimensions with unknown dimensions at compile time.
          // We then calculate the multidimensional array like this:
          // arr(x, y, z) bedcomes arr(z * stride(2) + y * stride(1) + x)
          // where stride is the distance between elements in the dimensions
          // 0, 1 and 2 or x, y and z.
          if (coop->getOperand(0) == arg.arg &&
              coop->getOperands().size() >= 2) {
            builder.setInsertionPoint(coop);
            mlir::Value totalIndex;
            for (unsigned i = arg.rank - 1; i > 0; i--) {
              mlir::Value curIndex =
                  builder.createConvert(loc, idxTy, getIndex(builder, coop, i));
              // Multiply by the stride of this array. Later we'll divide by
              // the element size.
              mlir::Value scale =
                  builder.createConvert(loc, idxTy, arg.dims[i].getResult(2));
              curIndex =
                  mlir::arith::MulIOp::create(builder, loc, scale, curIndex);
              totalIndex = (totalIndex)
                               ? mlir::arith::AddIOp::create(
                                     builder, loc, curIndex, totalIndex)
                               : curIndex;
            }
            // This is the lowest dimension - which doesn't need scaling
            mlir::Value finalIndex =
                builder.createConvert(loc, idxTy, getIndex(builder, coop, 0));
            if (totalIndex) {
              assert(llvm::isPowerOf2_32(arg.size) &&
                     "Expected power of two here");
              unsigned bits = llvm::Log2_32(arg.size);
              mlir::Value elemShift =
                  builder.createIntegerConstant(loc, idxTy, bits);
              totalIndex = mlir::arith::AddIOp::create(
                  builder, loc,
                  mlir::arith::ShRSIOp::create(builder, loc, totalIndex,
                                               elemShift),
                  finalIndex);
            } else {
              totalIndex = finalIndex;
            }
            auto newOp = fir::CoordinateOp::create(
                builder, loc, builder.getRefType(elementType), caddr,
                mlir::ValueRange{totalIndex});
            LLVM_DEBUG(newOp->dump());
            coop->getResult(0).replaceAllUsesWith(newOp->getResult(0));
            coop->erase();
            changed = true;
          }
        });

        builder.restoreInsertionPoint(insPt);
      }
    }
    assert(changed && "Expected operations to have changed");

    builder.insert(clonedLoop);
    // Forward the result(s), if any, from the loop operation to the
    //
    mlir::ResultRange results = clonedLoop->getResults();
    bool hasResults = (results.size() > 0);
    if (hasResults)
      fir::ResultOp::create(builder, loc, results);

    // Add the original loop in the else-side of the if operation.
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    op.op->replaceAllUsesWith(ifOp);
    op.op->remove();
    builder.insert(op.op);
    // Rely on "cloned loop has results, so original loop also has results".
    if (hasResults) {
      fir::ResultOp::create(builder, loc, op.op->getResults());
    } else {
      // Use an assert to check this.
      assert(op.op->getResults().size() == 0 &&
             "Weird, the cloned loop doesn't have results, but the original "
             "does?");
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Func After transform:\n");
  LLVM_DEBUG(func->dump());

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}
