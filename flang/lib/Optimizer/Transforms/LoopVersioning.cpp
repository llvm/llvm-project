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
/// As a side-effect of the assumed element size stride, the array is also
/// flattened to make it a 1D array - this is because the internal array
/// structure must be either 1D or have known sizes in all dimensions - and at
/// least one of the dimensions here is already unknown. Supported slices are
/// folded into the flattened indices.
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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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

/// @struct ArgInfo
/// A structure to hold an argument, the size of the argument and dimension
/// information.
struct ArgInfo {
  mlir::Value arg;
  size_t size;
  unsigned rank;
  fir::BoxDimsOp dims[CFI_MAX_RANK];
};

/// @struct ArgsUsageInLoop
/// A structure providing information about the function arguments
/// usage by the instructions immediately nested in a loop.
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
      for (auto &use : usageInfo) {
        mlir::Value v = use.first;
        v.print(llvm::dbgs(), printFlags);
        llvm::dbgs() << "\n";
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
      // A pointer dummy is a box address, but accesses use its loaded box.
      if (isArgument && fir::isBoxAddress(v.getType()))
        elementType = seqTy.getEleTy();
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

/// Follow descriptor producers to identify a function argument. Traverse a
/// load only when its source is a box address; other loads stop the walk.
/// The original access value is retained for the guard and rewritten loop.
static mlir::Value normaliseVal(mlir::Value val) {
  while (true) {
    if (fir::DeclareOp declare = val.getDefiningOp<fir::DeclareOp>()) {
      val = declare.getMemref();
      continue;
    }
    if (auto packArray = val.getDefiningOp<fir::PackArrayOp>()) {
      val = packArray.getArray();
      continue;
    }
    if (fir::ReboxOp rebox = val.getDefiningOp<fir::ReboxOp>()) {
      if (!fir::reboxPreservesContinuity(rebox,
                                         /*mayHaveNonDefaultLowerBounds=*/true,
                                         /*checkWhole=*/false)) {
        LLVM_DEBUG(llvm::dbgs() << "REBOX may produce non-contiguous array: "
                                << rebox << '\n');
        break;
      }
      val = rebox.getBox();
      continue;
    }
    if (fir::LoadOp load = val.getDefiningOp<fir::LoadOp>()) {
      if (!fir::isBoxAddress(load.getMemref().getType()))
        break;
      val = load.getMemref();
      continue;
    }
    break;
  }
  return val;
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

/// A fir.slice triple whose upper bound is fir.undefined selects a single
/// element instead of a section. XArrayCoor lowering then ignores the triple's
/// lower bound and step, and only uses the fir.array_coor index of that
/// dimension.
static bool isScalarSliceDim(mlir::ValueRange triples, unsigned dim) {
  return mlir::isa_and_nonnull<fir::UndefOp>(
      triples[3 * dim + 1].getDefiningOp());
}

/// Return the section lower bound that dimension \p dim of \p coop's slice
/// adds to the coordinate, or a null value if it adds nothing.
static mlir::Value getSliceLowerBound(fir::ArrayCoorOp coop, unsigned dim) {
  if (!coop.getSlice())
    return {};
  auto slice = mlir::cast<fir::SliceOp>(coop.getSlice().getDefiningOp());
  if (isScalarSliceDim(slice.getTriples(), dim))
    return {};
  return slice.getTriples()[3 * dim];
}

/// Match XArrayCoor's signed integerCast before computing sliced coordinates.
/// FIR conversion zero-extends builtin i1 and unsigned integers, whereas the
/// generic array-coordinate lowering sign-extends their bit patterns.
static mlir::Value toSliceIndex(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value value) {
  if (auto type = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
    if (type.getWidth() == 1) {
      value = builder.createConvert(loc, builder.getI1Type(), value);
      value = mlir::arith::ExtSIOp::create(builder, loc,
                                           builder.getIntegerType(2), value);
    } else if (type.isUnsigned()) {
      value = builder.createConvert(
          loc, builder.getIntegerType(type.getWidth()), value);
    }
  }
  if (value.getType().isSignlessInteger())
    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), value);
  return builder.createConvert(loc, builder.getIndexType(), value);
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
  if (coop.getSlice()) {
    // Convert a unit-step slice coordinate to a zero-based source-array index:
    // (index - lb) + (slice lower bound - lb); scalar dimensions omit the
    // adjustment.
    mlir::Location loc = coop.getLoc();
    index = toSliceIndex(builder, loc, index);
    lb = lb ? toSliceIndex(builder, loc, lb)
            : builder.createIntegerConstant(loc, builder.getIndexType(), 1);
    mlir::Value coor = mlir::arith::SubIOp::create(builder, loc, index, lb);
    if (mlir::Value sliceLb = getSliceLowerBound(coop, dim)) {
      sliceLb = toSliceIndex(builder, loc, sliceLb);
      mlir::Value adjust =
          builder.createOrFold<mlir::arith::SubIOp>(loc, sliceLb, lb);
      coor = builder.createOrFold<mlir::arith::AddIOp>(loc, coor, adjust);
    }
    return coor;
  }
  if (!lb)
    // assume a default lower bound of one
    lb = builder.createIntegerConstant(coop.getLoc(), index.getType(), 1);

  // index_0 = index - lb;
  if (lb.getType() != index.getType())
    lb = builder.createConvert(coop.getLoc(), index.getType(), lb);
  return mlir::arith::SubIOp::create(builder, coop.getLoc(), index, lb);
}

/// Return whether converting an integer through \p type may turn the value one
/// into something else. A one-bit builtin or kind-mapped FIR integer denotes
/// -1 once XArrayCoor lowering sign extends it again.
static bool mayNotPreserveOne(mlir::Type type,
                              const fir::KindMapping &kindMap) {
  if (mlir::isa<mlir::IndexType>(type))
    return false;
  if (auto intTy = mlir::dyn_cast<fir::IntegerType>(type))
    return kindMap.getIntegerBitsize(intTy.getFKind()) <= 1;
  auto intTy = mlir::dyn_cast<mlir::IntegerType>(type);
  return !intTy || intTy.getWidth() <= 1;
}

/// Return whether \p value is the constant one, looking through fir.convert.
static bool isConstantOne(mlir::Value value, const fir::KindMapping &kindMap) {
  while (auto convert = value.getDefiningOp<fir::ConvertOp>()) {
    if (mayNotPreserveOne(convert.getType(), kindMap))
      return false;
    value = convert.getValue();
  }
  if (mayNotPreserveOne(value.getType(), kindMap))
    return false;
  std::optional<llvm::APInt> constant = fir::getIntIfConstant(value);
  return constant && constant->isOne();
}

/// Return whether the slice of \p coop can be folded into the flat index
/// computed for the fast loop version.
///
/// XArrayCoor lowering computes the zero based coordinate of dimension i of a
/// boxed array as `(index - lb) * step + (sliceLb - lb)`, where step and
/// sliceLb only contribute for a section. With a unit step this is the
/// slice-free coordinate `index - lb` plus the loop invariant
/// `sliceLb - lb`, so flattening only needs that extra term. Anything that
/// does not fit is left on the generic path.
static bool isFoldableSlice(fir::ArrayCoorOp coop, unsigned rank,
                            const fir::KindMapping &kindMap) {
  auto slice = coop.getSlice().getDefiningOp<fir::SliceOp>();
  // ArrayCoorOp's verifier rejects slices with substring operands, so they do
  // not need to be checked here.
  if (!slice || !slice.getFields().empty())
    return false;

  mlir::ValueRange triples = slice.getTriples();
  // Reduced-rank slices need a separate mapping from result coordinates to
  // source dimensions. The current flattening handles only one coordinate per
  // descriptor dimension.
  if (coop.getIndices().size() != rank)
    return false;

  // TODO: Support the remaining valid fir.slice forms. This initial slice
  // implementation leaves non-unit section steps on the generic path.
  for (unsigned dim = 0; dim < rank; ++dim) {
    if (isScalarSliceDim(triples, dim))
      continue;
    mlir::Value lower = triples[3 * dim];
    mlir::Value step = triples[3 * dim + 2];
    if (mlir::isa_and_nonnull<fir::UndefOp>(lower.getDefiningOp()) ||
        mlir::isa_and_nonnull<fir::UndefOp>(step.getDefiningOp()))
      return false;
    // TODO: Fold constant zero and one through fir.convert when motivating
    // source cases justify changing the dialect-wide folder.
    if (!isConstantOne(step, kindMap))
      return false;
  }
  return true;
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
      argsOfInterest.push_back({arg, typeSize, rank, {}});
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
  llvm::DenseMap<fir::DoLoopOp, ArgsUsageInLoop> argsInLoops;

  auto &domInfo = getAnalysis<mlir::DominanceInfo>();

  // Traverse the loops in post-order and see
  // if those arguments are used inside any loop.
  func.walk([&](fir::DoLoopOp loop) {
    mlir::Block &body = *loop.getBody();
    auto &argsInLoop = argsInLoops[loop];
    originalLoops.push_back(loop);
    body.walk([&](mlir::Operation *op) {
      // Support either fir.array_coor or fir.coordinate_of.
      if (!mlir::isa<fir::ArrayCoorOp, fir::CoordinateOp>(op))
        return;
      // Process only operations immediately nested in the current loop.
      if (op->getParentOfType<fir::DoLoopOp>() != loop)
        return;
      mlir::Value operand = op->getOperand(0);
      mlir::Value source = normaliseVal(operand);
      for (auto a : argsOfInterest) {
        if (a.arg == source) {
          // Use the access descriptor, not its originating block argument.
          a.arg = operand;

          // A rejected descriptor cannot recover in a later access. Skip
          // repeated dominance, type, and slice analysis for such accesses.
          bool rejected = argsInLoop.cannotTransform.contains(a.arg);
          if (!rejected) {
            rejected = !domInfo.dominates(a.arg, loop);

            // Compute rank and element size from the access descriptor rather
            // than the original argument, since intervening descriptor
            // operations may change them.
            if (!rejected) {
              std::tie(a.rank, a.size) =
                  getRankAndElementSize(kindMap, *dl, a.arg);
              rejected = a.rank == 0 || a.size == 0;

              // A slice must fold into the flat index; otherwise this
              // descriptor and its later accesses stay on the generic path.
              if (!rejected) {
                if (auto arrayCoor = mlir::dyn_cast<fir::ArrayCoorOp>(op);
                    arrayCoor && arrayCoor.getSlice())
                  rejected = !enableSlices ||
                             !isFoldableSlice(arrayCoor, a.rank, kindMap);
              }
            }
          }

          if (rejected) {
            argsInLoop.cannotTransform.insert(a.arg);
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
  });

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
  };
  mlir::SmallVector<OpsWithArgs, 4> loopsOfInterest;
  for (fir::DoLoopOp loop : originalLoops) {
    auto &argsInLoop = argsInLoops[loop];
    if (argsInLoop.usageInfo.empty())
      continue;
    OpsWithArgs info;
    info.op = loop;
    for (auto &arg : argsInLoop.usageInfo)
      info.argsAndDims.push_back(arg.second);
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
    builder.setInsertionPoint(op.op);

    mlir::Value allCompares = nullptr;
    // Ensure all of the arrays are unit-stride.
    for (auto &arg : op.argsAndDims) {
      // Fetch all the dimensions of the array, except the last dimension.
      // Always fetch the first dimension, however, so set ndims = 1 if
      // we have one dim
      unsigned ndims = arg.rank;
      for (unsigned i = 0; i < ndims; i++) {
        mlir::Value dimIdx = builder.createIntegerConstant(loc, idxTy, i);
        arg.dims[i] = fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy,
                                             arg.arg, dimIdx);
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

    auto ifOp =
        fir::IfOp::create(builder, loc, op.op->getResultTypes(), allCompares,
                          /*withElse=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    LLVM_DEBUG(llvm::dbgs() << "Creating cloned loop\n");
    mlir::Operation *clonedLoop = op.op->clone();
    bool changed = false;
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
        // This is required becase fir arrays do not support multiple dimensions
        // with unknown dimensions at compile time.
        // We then calculate the multidimensional array like this:
        // arr(x, y, z) bedcomes arr(z * stride(2) + y * stride(1) + x)
        // where stride is the distance between elements in the dimensions
        // 0, 1 and 2 or x, y and z.
        if (coop->getOperand(0) == arg.arg && coop->getOperands().size() >= 2) {
          builder.setInsertionPoint(coop);
          mlir::Value totalIndex;
          for (unsigned i = arg.rank - 1; i > 0; i--) {
            mlir::Value curIndex =
                builder.createConvert(loc, idxTy, getIndex(builder, coop, i));
            // Multiply by the stride of this array. Later we'll divide by the
            // element size.
            mlir::Value scale =
                builder.createConvert(loc, idxTy, arg.dims[i].getResult(2));
            curIndex =
                mlir::arith::MulIOp::create(builder, loc, scale, curIndex);
            totalIndex = (totalIndex) ? mlir::arith::AddIOp::create(
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
