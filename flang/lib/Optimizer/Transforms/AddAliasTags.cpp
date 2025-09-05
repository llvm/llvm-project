//===- AddAliasTags.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// Adds TBAA alias tags to fir loads and stores, based on information from
/// fir::AliasAnalysis. More are added later in CodeGen - see fir::TBAABuilder
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Analysis/TBAAForest.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace fir {
#define GEN_PASS_DEF_ADDALIASTAGS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "fir-add-alias-tags"

static llvm::cl::opt<bool>
    enableDummyArgs("dummy-arg-tbaa", llvm::cl::init(true), llvm::cl::Hidden,
                    llvm::cl::desc("Add TBAA tags to dummy arguments"));
static llvm::cl::opt<bool>
    enableGlobals("globals-tbaa", llvm::cl::init(true), llvm::cl::Hidden,
                  llvm::cl::desc("Add TBAA tags to global variables"));
static llvm::cl::opt<bool>
    enableDirect("direct-tbaa", llvm::cl::init(true), llvm::cl::Hidden,
                 llvm::cl::desc("Add TBAA tags to direct variables"));
static llvm::cl::opt<bool>
    enableLocalAllocs("local-alloc-tbaa", llvm::cl::init(true),
                      llvm::cl::Hidden,
                      llvm::cl::desc("Add TBAA tags to local allocations."));

// Engineering option to triage TBAA tags attachment for accesses
// of allocatable entities.
static llvm::cl::opt<unsigned> localAllocsThreshold(
    "local-alloc-tbaa-threshold", llvm::cl::init(0), llvm::cl::ReallyHidden,
    llvm::cl::desc("If present, stops generating TBAA tags for accesses of "
                   "local allocations after N accesses in a module"));

namespace {

// Return the size and alignment (in bytes) for the given type.
// TODO: this must be combined with DebugTypeGenerator::getFieldSizeAndAlign().
// We'd better move fir::LLVMTypeConverter out of the FIRCodeGen component.
static std::pair<std::uint64_t, unsigned short>
getTypeSizeAndAlignment(mlir::Type type,
                        fir::LLVMTypeConverter &llvmTypeConverter) {
  mlir::Type llvmTy;
  if (auto boxTy = mlir::dyn_cast_if_present<fir::BaseBoxType>(type))
    llvmTy = llvmTypeConverter.convertBoxTypeAsStruct(boxTy, getBoxRank(boxTy));
  else
    llvmTy = llvmTypeConverter.convertType(type);

  const mlir::DataLayout &dataLayout = llvmTypeConverter.getDataLayout();
  uint64_t byteSize = dataLayout.getTypeSize(llvmTy);
  unsigned short byteAlign = dataLayout.getTypeABIAlignment(llvmTy);
  return std::pair{byteSize, byteAlign};
}

// IntervalTy class describes a range of bytes addressed by a variable
// within some storage. Zero-sized intervals are not allowed.
class IntervalTy {
public:
  IntervalTy() = delete;
  IntervalTy(std::uint64_t start, std::size_t size)
      : start(start), end(start + (size - 1)) {
    assert(size != 0 && "empty intervals should not be created");
  }
  constexpr bool operator<(const IntervalTy &rhs) const {
    if (start < rhs.start)
      return true;
    if (rhs.start < start)
      return false;
    return end < rhs.end;
  }
  bool overlaps(const IntervalTy &other) const {
    return end >= other.start && other.end >= start;
  }
  bool contains(const IntervalTy &other) const {
    return start <= other.start && end >= other.end;
  }
  void merge(const IntervalTy &other) {
    start = std::min(start, other.start);
    end = std::max(end, other.end);
    assert(start <= end);
  }
  void print(llvm::raw_ostream &os) const {
    os << "[" << start << "," << end << "]";
  }
  std::uint64_t getStart() const { return start; }
  std::uint64_t getEnd() const { return end; }

private:
  std::uint64_t start;
  std::uint64_t end;
};

// IntervalSetTy is an ordered set of IntervalTy entities.
class IntervalSetTy : public std::set<IntervalTy> {
public:
  // Find an interval from the set that contain the given interval.
  // The complexity is O(log(N)), where N is the size of the set.
  std::optional<IntervalTy> getContainingInterval(const IntervalTy &interval) {
    if (empty())
      return std::nullopt;

    auto it = lower_bound(interval);
    // The iterator points to the first interval that is not less than
    // the given interval. The given interval may belong to the one
    // pointed out by the iterator or to the previous one.
    //
    // In the following cases there might be no interval that is not less
    // than the given interval, e.g.:
    // Case 1:
    //   interval: [5,5]
    //   set: {[4,6]}
    // Case 2:
    //   interval: [5,5]
    //   set: {[4,5]}
    // We have to look starting from the last interval in the set.
    if (it == end())
      --it;

    // The loop must finish in two iterator max.
    do {
      if (it->contains(interval))
        return *it;
      // If the current interval from the set is less than the given
      // interval and there is no overlap, we should not look further.
      if ((!it->overlaps(interval) && *it < interval) || it == begin())
        break;

      --it;
    } while (true);

    return std::nullopt;
  }
};

// Stream operators for IntervalTy and IntervalSetTy.
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const IntervalTy &interval) {
  interval.print(os);
  return os;
}

[[maybe_unused]] inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const IntervalSetTy &set) {
  if (set.empty()) {
    os << " <empty>";
    return os;
  }
  for (const auto &interval : set)
    os << ' ' << interval;
  return os;
}

/// Shared state per-module
class PassState {
public:
  PassState(mlir::ModuleOp module, const mlir::DataLayout &dl,
            mlir::DominanceInfo &domInfo,
            std::optional<unsigned> localAllocsThreshold)
      : domInfo(domInfo), localAllocsThreshold(localAllocsThreshold),
        symTab(module.getOperation()),
        llvmTypeConverter(module, /*applyTBAA=*/false,
                          /*forceUnifiedTBAATree=*/false, dl) {}
  /// memoised call to fir::AliasAnalysis::getSource
  inline const fir::AliasAnalysis::Source &getSource(mlir::Value value) {
    if (!analysisCache.contains(value))
      analysisCache.insert(
          {value, analysis.getSource(value, /*getInstantiationPoint=*/true)});
    return analysisCache[value];
  }

  /// get the per-function TBAATree for this function
  inline fir::TBAATree &getMutableFuncTreeWithScope(mlir::func::FuncOp func,
                                                    fir::DummyScopeOp scope) {
    auto &scopeMap = scopeNames.at(func);
    return forrest.getMutableFuncTreeWithScope(func, scopeMap.lookup(scope));
  }
  inline const fir::TBAATree &getFuncTreeWithScope(mlir::func::FuncOp func,
                                                   fir::DummyScopeOp scope) {
    return getMutableFuncTreeWithScope(func, scope);
  }

  void processFunctionScopes(mlir::func::FuncOp func);
  // For the given fir.declare returns the dominating fir.dummy_scope
  // operation.
  fir::DummyScopeOp getDeclarationScope(fir::DeclareOp declareOp) const;
  // For the given fir.declare returns the outermost fir.dummy_scope
  // in the current function.
  fir::DummyScopeOp getOutermostScope(fir::DeclareOp declareOp) const;
  // Returns true, if the given type of a memref of a FirAliasTagOpInterface
  // operation is a descriptor or contains a descriptor
  // (e.g. !fir.ref<!fir.type<Derived{f:!fir.box<!fir.heap<f32>>}>>).
  bool typeReferencesDescriptor(mlir::Type type);

  // Returns true if we can attach a TBAA tag to an access of an allocatable
  // entities. It checks if localAllocsThreshold allows the next tag
  // attachment.
  bool attachLocalAllocTag();

  // Return fir.global for the given name.
  fir::GlobalOp getGlobalDefiningOp(mlir::StringAttr name) const {
    return symTab.lookup<fir::GlobalOp>(name);
  }

  // Process fir::FortranVariableStorageOpInterface operations within
  // the given op, and fill in declToStorageMap with the information
  // about their physical storages and layouts.
  void collectPhysicalStorageAliasSets(mlir::Operation *op);

  // Return the byte size of the given declaration.
  std::size_t getDeclarationSize(fir::FortranVariableStorageOpInterface decl) {
    mlir::Type memType = fir::unwrapRefType(decl.getBase().getType());
    auto [size, alignment] =
        getTypeSizeAndAlignment(memType, llvmTypeConverter);
    return llvm::alignTo(size, alignment);
  }

  // A StorageDesc specifies an operation that defines a physical storage
  // and the <offset, size> pair within that physical storage where
  // a variable resides.
  struct StorageDesc {
    StorageDesc() = delete;
    StorageDesc(mlir::Operation *storageDef, std::uint64_t start,
                std::size_t size)
        : storageDef(storageDef), interval(start, size) {}

    // Return a string representing the byte range of the variable within
    // its storage, e.g. bytes_0_to_0 for a 1-byte variable starting
    // at offset 0.
    std::string getByteRangeStr() const {
      return ("bytes_" + llvm::Twine(interval.getStart()) + "_to_" +
              llvm::Twine(interval.getEnd()))
          .str();
    }

    mlir::Operation *storageDef;
    IntervalTy interval;
  };

  // Fills in declToStorageMap on the first invocation.
  // Returns a storage descriptor for the given op (if registered
  // in declToStorageMap).
  const StorageDesc *computeStorageDesc(mlir::Operation *op) {
    if (!op)
      return nullptr;

    // TODO: it should be safe to run collectPhysicalStorageAliasSets()
    // on the parent func.func instead of the module, since the TBAA
    // tags use different roots per function. This may provide better
    // results for storages that have members with descriptors
    // in one function but not the others.
    if (!declToStorageMapComputed)
      collectPhysicalStorageAliasSets(op->getParentOfType<mlir::ModuleOp>());
    return getStorageDesc(op);
  }

private:
  const StorageDesc *getStorageDesc(mlir::Operation *op) const {
    auto it = declToStorageMap.find(op);
    return it == declToStorageMap.end() ? nullptr : &it->second;
  }

  StorageDesc &getMutableStorageDesc(mlir::Operation *op) {
    auto it = declToStorageMap.find(op);
    assert(it != declToStorageMap.end());
    return it->second;
  }

private:
  mlir::DominanceInfo &domInfo;
  std::optional<unsigned> localAllocsThreshold;
  // Symbol table cache for the module.
  mlir::SymbolTable symTab;
  // Type converter to compute the size of declarations.
  fir::LLVMTypeConverter llvmTypeConverter;
  fir::AliasAnalysis analysis;
  llvm::DenseMap<mlir::Value, fir::AliasAnalysis::Source> analysisCache;
  fir::TBAAForrest forrest;
  // Unique names for fir.dummy_scope operations within
  // the given function.
  llvm::DenseMap<mlir::func::FuncOp,
                 llvm::DenseMap<fir::DummyScopeOp, std::string>>
      scopeNames;
  // A map providing a vector of fir.dummy_scope operations
  // for the given function. The vectors are sorted according
  // to the dominance information.
  llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<fir::DummyScopeOp, 16>>
      sortedScopeOperations;

  // Local pass cache for derived types that contain descriptor
  // member(s), to avoid the cost of isRecordWithDescriptorMember().
  llvm::DenseSet<mlir::Type> typesContainingDescriptors;

  // A map between fir::FortranVariableStorageOpInterface operations
  // and their storage descriptors.
  llvm::DenseMap<mlir::Operation *, StorageDesc> declToStorageMap;
  // declToStorageMapComputed is set to true after declToStorageMap
  // is initialized by collectPhysicalStorageAliasSets().
  bool declToStorageMapComputed = false;
};

// Process fir.dummy_scope operations in the given func:
// sort them according to the dominance information, and
// associate a unique (within the current function) scope name
// with each of them.
void PassState::processFunctionScopes(mlir::func::FuncOp func) {
  if (scopeNames.contains(func))
    return;

  auto &scopeMap = scopeNames[func];
  auto &scopeOps = sortedScopeOperations[func];
  func.walk([&](fir::DummyScopeOp op) { scopeOps.push_back(op); });
  llvm::stable_sort(scopeOps, [&](const fir::DummyScopeOp &op1,
                                  const fir::DummyScopeOp &op2) {
    return domInfo.properlyDominates(&*op1, &*op2);
  });
  unsigned scopeId = 0;
  for (auto scope : scopeOps) {
    if (scopeId != 0) {
      std::string name = (llvm::Twine("Scope ") + llvm::Twine(scopeId)).str();
      LLVM_DEBUG(llvm::dbgs() << "Creating scope '" << name << "':\n"
                              << scope << "\n");
      scopeMap.insert({scope, std::move(name)});
    }
    ++scopeId;
  }
}

fir::DummyScopeOp
PassState::getDeclarationScope(fir::DeclareOp declareOp) const {
  auto func = declareOp->getParentOfType<mlir::func::FuncOp>();
  assert(func && "fir.declare does not have parent func.func");
  auto &scopeOps = sortedScopeOperations.at(func);
  for (auto II = scopeOps.rbegin(), IE = scopeOps.rend(); II != IE; ++II) {
    if (domInfo.dominates(&**II, &*declareOp))
      return *II;
  }
  return nullptr;
}

fir::DummyScopeOp PassState::getOutermostScope(fir::DeclareOp declareOp) const {
  auto func = declareOp->getParentOfType<mlir::func::FuncOp>();
  assert(func && "fir.declare does not have parent func.func");
  auto &scopeOps = sortedScopeOperations.at(func);
  if (!scopeOps.empty())
    return scopeOps[0];
  return nullptr;
}

bool PassState::typeReferencesDescriptor(mlir::Type type) {
  type = fir::unwrapAllRefAndSeqType(type);
  if (mlir::isa<fir::BaseBoxType>(type))
    return true;

  if (mlir::isa<fir::RecordType>(type)) {
    if (typesContainingDescriptors.contains(type))
      return true;
    if (fir::isRecordWithDescriptorMember(type)) {
      typesContainingDescriptors.insert(type);
      return true;
    }
  }
  return false;
}

bool PassState::attachLocalAllocTag() {
  if (!localAllocsThreshold)
    return true;
  if (*localAllocsThreshold == 0) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "WARN: not assigning TBAA tag for an allocated entity access "
                  "due to the threshold\n");
    return false;
  }
  --*localAllocsThreshold;
  return true;
}

static mlir::Value getStorageDefinition(mlir::Value storageRef) {
  while (auto convert =
             mlir::dyn_cast_or_null<fir::ConvertOp>(storageRef.getDefiningOp()))
    storageRef = convert.getValue();
  return storageRef;
}

void PassState::collectPhysicalStorageAliasSets(mlir::Operation *op) {
  // A map between fir::FortranVariableStorageOpInterface operations
  // and the intervals describing their layout within their physical
  // storages.
  llvm::DenseMap<mlir::Operation *, IntervalSetTy> memberIntervals;
  // A map between operations defining physical storages (e.g. fir.global)
  // and sets of fir::FortranVariableStorageOpInterface operations
  // declaring their member variables.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 10>>
      storageDecls;

  bool seenUnknownStorage = false;
  bool seenDeclWithDescriptor = false;
  op->walk([&](fir::FortranVariableStorageOpInterface decl) {
    mlir::Value storageRef = decl.getStorage();
    if (!storageRef)
      return mlir::WalkResult::advance();

    // If we have seen a declaration of a variable containing
    // a descriptor, and we have not been able to identify
    // a storage of any variable, then any variable may
    // potentially overlap with the variable containing
    // a descriptor. In this case, it is hard to make any
    // assumptions about any variable with physical
    // storage. Exit early.
    if (seenUnknownStorage && seenDeclWithDescriptor)
      return mlir::WalkResult::interrupt();

    if (typeReferencesDescriptor(decl.getBase().getType()))
      seenDeclWithDescriptor = true;

    mlir::Operation *storageDef =
        getStorageDefinition(storageRef).getDefiningOp();
    // All physical storages that are defined by non-global
    // objects (e.g. via fir.alloca) indicate an EQUIVALENCE.
    // Inside an EQUIVALENCE each variable overlaps
    // with at least one another variable. So all EQUIVALENCE
    // variables belong to the same alias set, and there is
    // no reason to investigate them further.
    // Note that, in general, the storage may be defined by a block
    // argument.
    auto addrOfOp = mlir::dyn_cast_or_null<fir::AddrOfOp>(storageDef);
    if (!storageDef ||
        (!addrOfOp && !mlir::dyn_cast<fir::AllocaOp>(storageDef))) {
      seenUnknownStorage = true;
      return mlir::WalkResult::advance();
    }
    if (!addrOfOp)
      return mlir::WalkResult::advance();
    fir::GlobalOp globalDef =
        getGlobalDefiningOp(addrOfOp.getSymbol().getRootReference());
    std::uint64_t storageOffset = decl.getStorageOffset();
    std::size_t declSize = getDeclarationSize(decl);
    LLVM_DEBUG(llvm::dbgs()
               << "Found variable with storage:\n"
               << "Declaration: " << decl << "\n"
               << "Storage: " << (globalDef ? globalDef : nullptr) << "\n"
               << "Offset: " << storageOffset << "\n"
               << "Size: " << declSize << "\n");
    if (!globalDef) {
      seenUnknownStorage = true;
      return mlir::WalkResult::advance();
    }
    // Zero-sized variables do not need any TBAA tags, because
    // they cannot be accessed.
    if (declSize == 0)
      return mlir::WalkResult::advance();

    declToStorageMap.try_emplace(decl.getOperation(), globalDef.getOperation(),
                                 storageOffset, declSize);
    storageDecls.try_emplace(globalDef.getOperation())
        .first->second.push_back(decl.getOperation());

    auto &set =
        memberIntervals.try_emplace(globalDef.getOperation()).first->second;
    set.insert(IntervalTy(storageOffset, declSize));
    return mlir::WalkResult::advance();
  });

  // Mark the map as computed before any early exits below.
  declToStorageMapComputed = true;

  if (seenUnknownStorage && seenDeclWithDescriptor) {
    declToStorageMap.clear();
    return;
  }

  // Process each physical storage.
  for (auto &map : memberIntervals) {
    mlir::Operation *storageDef = map.first;
    const IntervalSetTy &originalSet = map.second;
    LLVM_DEBUG(
        llvm::dbgs() << "Merging " << originalSet.size()
                     << " member intervals for: ";
        storageDef->print(llvm::dbgs(), mlir::OpPrintingFlags{}.skipRegions());
        llvm::dbgs() << "\nIntervals: " << originalSet << "\n");
    // Ordered set of merged overlapping intervals.
    // Since the intervals in originalSet are sorted, the merged
    // intervals are always added at the end of the mergedIntervals set.
    IntervalSetTy mergedIntervals;
    if (originalSet.size() > 1) {
      auto intervalIt = originalSet.begin();
      IntervalTy mergedInterval = *intervalIt;
      while (++intervalIt != originalSet.end()) {
        if (mergedInterval.overlaps(*intervalIt)) {
          mergedInterval.merge(*intervalIt);
        } else {
          mergedIntervals.insert(mergedIntervals.end(), mergedInterval);
          mergedInterval = *intervalIt;
        }
      }
      mergedIntervals.insert(mergedIntervals.end(), mergedInterval);
    } else {
      // 0 or 1 total interval requires no merging.
      mergedIntervals = originalSet;
    }
    LLVM_DEBUG(llvm::dbgs() << "Merged intervals:" << mergedIntervals << "\n");

    bool wasMerged = originalSet.size() != mergedIntervals.size();

    // Go through all the declarations within the storage, and assign
    // them to their final intervals (if some merging happened),
    // and collect information about "poisoned" intervals (see below).
    // invalidIntervals set will contain the "poisoned" intervals.
    IntervalSetTy invalidIntervals;
    for (auto *decl : storageDecls.at(storageDef)) {
      StorageDesc &declStorageDesc = getMutableStorageDesc(decl);

      if (wasMerged) {
        // Some intervals were merged, so we have to modify the intervals
        // for some declarations.

        auto containingInterval =
            mergedIntervals.getContainingInterval(declStorageDesc.interval);
        assert(containingInterval && "did not find the containing interval");
        LLVM_DEBUG(llvm::dbgs() << "Placing: " << *decl << " into interval "
                                << *containingInterval);
        declStorageDesc.interval = *containingInterval;
      }
      if (typeReferencesDescriptor(
              mlir::cast<fir::FortranVariableStorageOpInterface>(decl)
                  .getBase()
                  .getType())) {
        // If a variable contains a descriptor within it.
        // We cannot attach any data tag to it, because it will
        // conflict with the late TBBA tags attachment for
        // the descriptor data. This also applies to all
        // variables overlapping with this one, thus we should
        // remove any storage descriptors for their declarations.
        LLVM_DEBUG(llvm::dbgs() << " (poisoned)");
        invalidIntervals.insert(declStorageDesc.interval);
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }

    if (invalidIntervals.empty())
      continue;

    // Now that all the declarations are assigned to their intervals,
    // go through the "poisoned" intervals and remove all declarations
    // belonging to them from declToStorageMap, so that they do not
    // have any tags attached.
    LLVM_DEBUG(llvm::dbgs()
               << "Invalid intervals:" << invalidIntervals << "\n");
    if (invalidIntervals.size() == mergedIntervals.size()) {
      // All variables are "poisoned". Save the O(log(N)) lookups
      // in invalidIntervals set, and poison them all.
      for (auto *decl : storageDecls.at(storageDef)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Removing storage descriptor for: " << *decl << "\n");
        declToStorageMap.erase(decl);
      }
      continue;
    }

    // Some variables are "poisoned".
    for (auto *decl : storageDecls.at(storageDef)) {
      const StorageDesc *declStorageDesc = getStorageDesc(decl);
      assert(declStorageDesc && "declaration must have a storage descriptor");
      if (auto containingInterval = invalidIntervals.getContainingInterval(
              declStorageDesc->interval)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Removing storage descriptor for: " << *decl << "\n");
        declToStorageMap.erase(decl);
      }
    }
  }
}

class AddAliasTagsPass : public fir::impl::AddAliasTagsBase<AddAliasTagsPass> {
public:
  void runOnOperation() override;

private:
  /// The real workhorse of the pass. This is a runOnOperation() which
  /// operates on fir::FirAliasTagOpInterface, using some extra state
  void runOnAliasInterface(fir::FirAliasTagOpInterface op, PassState &state);
};

} // namespace

static fir::DeclareOp getDeclareOp(mlir::Value arg) {
  if (auto declare =
          mlir::dyn_cast_or_null<fir::DeclareOp>(arg.getDefiningOp()))
    return declare;
  for (mlir::Operation *use : arg.getUsers())
    if (fir::DeclareOp declare = mlir::dyn_cast<fir::DeclareOp>(use))
      return declare;
  return nullptr;
}

/// Get the name of a function argument using the "fir.bindc_name" attribute,
/// or ""
static std::string getFuncArgName(mlir::Value arg) {
  // first try getting the name from the fir.declare
  if (fir::DeclareOp declare = getDeclareOp(arg))
    return declare.getUniqName().str();

  // get from attribute on function argument
  // always succeeds because arg is a function argument
  mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(arg);
  assert(blockArg.getOwner() && blockArg.getOwner()->isEntryBlock() &&
         "arg is a function argument");
  mlir::FunctionOpInterface func = mlir::dyn_cast<mlir::FunctionOpInterface>(
      blockArg.getOwner()->getParentOp());
  assert(func && "This is not a function argument");
  mlir::StringAttr attr = func.getArgAttrOfType<mlir::StringAttr>(
      blockArg.getArgNumber(), "fir.bindc_name");
  if (!attr)
    return "";
  return attr.str();
}

void AddAliasTagsPass::runOnAliasInterface(fir::FirAliasTagOpInterface op,
                                           PassState &state) {
  mlir::func::FuncOp func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return;

  llvm::SmallVector<mlir::Value> accessedOperands = op.getAccessedOperands();
  assert(accessedOperands.size() == 1 &&
         "load and store only access one address");
  mlir::Value memref = accessedOperands.front();

  // Skip boxes and derived types that contain descriptors.
  // The box accesses get an "any descriptor access" tag in TBAABuilder
  // (CodeGen). The derived types accesses get "any access" tag
  // (because they access both the data and the descriptor(s)).
  // Note that it would be incorrect to attach any "data" access
  // tag to the derived type accesses here, because the tags
  // attached to the descriptor accesses in CodeGen will make
  // them non-conflicting with any descriptor accesses.
  if (state.typeReferencesDescriptor(memref.getType()))
    return;

  LLVM_DEBUG(llvm::dbgs() << "Analysing " << op << "\n");

  const fir::AliasAnalysis::Source &source = state.getSource(memref);

  // Process the scopes, if not processed yet.
  state.processFunctionScopes(func);

  fir::DummyScopeOp scopeOp;
  if (auto declOp = source.origin.instantiationPoint) {
    // If the source is a dummy argument within some fir.dummy_scope,
    // then find the corresponding innermost scope to be used for finding
    // the right TBAA tree.
    auto declareOp = mlir::dyn_cast<fir::DeclareOp>(declOp);
    assert(declareOp && "Instantiation point must be fir.declare");
    if (auto dummyScope = declareOp.getDummyScope())
      scopeOp = mlir::cast<fir::DummyScopeOp>(dummyScope.getDefiningOp());
    if (!scopeOp)
      scopeOp = state.getDeclarationScope(declareOp);
  }

  mlir::LLVM::TBAATagAttr tag;
  // TBAA for dummy arguments
  if (enableDummyArgs &&
      source.kind == fir::AliasAnalysis::SourceKind::Argument) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Found reference to dummy argument at " << *op << "\n");
    std::string name = getFuncArgName(llvm::cast<mlir::Value>(source.origin.u));
    // If it is a TARGET or POINTER, then we do not care about the name,
    // because the tag points to the root of the subtree currently.
    if (source.isTargetOrPointer()) {
      tag = state.getFuncTreeWithScope(func, scopeOp).targetDataTree.getTag();
    } else if (!name.empty()) {
      tag = state.getFuncTreeWithScope(func, scopeOp)
                .dummyArgDataTree.getTag(name);
    } else {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "WARN: couldn't find a name for dummy argument " << *op
                 << "\n");
      tag = state.getFuncTreeWithScope(func, scopeOp).dummyArgDataTree.getTag();
    }

    // TBAA for global variables without descriptors
  } else if (enableGlobals &&
             source.kind == fir::AliasAnalysis::SourceKind::Global &&
             !source.isBoxData()) {
    mlir::SymbolRefAttr glbl = llvm::cast<mlir::SymbolRefAttr>(source.origin.u);
    mlir::StringAttr globalName = glbl.getRootReference();
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Found reference to global " << globalName.str() << " at "
               << *op << "\n");
    if (source.isPointer()) {
      tag = state.getFuncTreeWithScope(func, scopeOp).targetDataTree.getTag();
    } else {
      // In general, place the tags under the "global data" root.
      fir::TBAATree::SubtreeState *subTree =
          &state.getMutableFuncTreeWithScope(func, scopeOp).globalDataTree;

      mlir::Operation *instantiationPoint = source.origin.instantiationPoint;
      auto storageIface =
          mlir::dyn_cast_or_null<fir::FortranVariableStorageOpInterface>(
              instantiationPoint);
      const PassState::StorageDesc *storageDesc =
          state.computeStorageDesc(instantiationPoint);

      if (storageDesc) {
        // This is a variable that is part of a known physical storage
        // that may contain multiple and maybe overlapping variables.
        // We have may assign it with a tag that relates
        // to the byte range within the physical storage.
        assert(instantiationPoint && "cannot be null");
        assert(storageDesc->storageDef && "cannot be null");
        assert(storageDesc->storageDef ==
                   state.getGlobalDefiningOp(globalName) &&
               "alias analysis reached a different storage");
        std::string aliasSetName = storageDesc->getByteRangeStr();
        subTree = &subTree->getOrCreateNamedSubtree(globalName);
        tag = subTree->getTag(aliasSetName);
        LLVM_DEBUG(llvm::dbgs()
                   << "Variable instantiated by " << *instantiationPoint
                   << " tagged with '" << aliasSetName << "' under '"
                   << globalName << "' root\n");
      } else if (storageIface && storageIface.getStorage()) {
        // This is a variable that is:
        //   * aliasing a descriptor, or
        //   * part of an unknown physical storage, or
        //   * zero-sized.
        // If it aliases a descriptor or the storage is unknown
        // (i.e. it *may* alias a descriptor), then we cannot assign any tag to
        // it, because we cannot use any tag from the "any data accesses" tree.
        // If it is a zero-sized variable, we do not care about
        // attaching a tag, because the access is invalid.
        LLVM_DEBUG(llvm::dbgs() << "WARNING: poisoned or unknown storage or "
                                   "zero-sized variable access\n");
      } else {
        // This is a variable defined by the global symbol,
        // and it is the only variable that belong to that global storage.
        // Tag it using the global's name.
        tag = subTree->getTag(globalName);
        LLVM_DEBUG(llvm::dbgs()
                   << "Tagged under '" << globalName << "' root\n");
      }
    }

    // TBAA for global variables with descriptors
  } else if (enableDirect &&
             source.kind == fir::AliasAnalysis::SourceKind::Global &&
             source.isBoxData()) {
    if (auto glbl = llvm::dyn_cast<mlir::SymbolRefAttr>(source.origin.u)) {
      const char *name = glbl.getRootReference().data();
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Found reference to direct " << name
                                        << " at " << *op << "\n");
      if (source.isPointer())
        tag = state.getFuncTreeWithScope(func, scopeOp).targetDataTree.getTag();
      else
        tag = state.getFuncTreeWithScope(func, scopeOp)
                  .directDataTree.getTag(name);
    } else {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Can't get name for direct "
                                        << source << " at " << *op << "\n");
    }

    // TBAA for local allocations
  } else if (enableLocalAllocs &&
             source.kind == fir::AliasAnalysis::SourceKind::Allocate) {
    std::optional<llvm::StringRef> name;
    mlir::Operation *sourceOp =
        llvm::cast<mlir::Value>(source.origin.u).getDefiningOp();
    bool unknownAllocOp = false;
    if (auto alloc = mlir::dyn_cast_or_null<fir::AllocaOp>(sourceOp))
      name = alloc.getUniqName();
    else if (auto alloc = mlir::dyn_cast_or_null<fir::AllocMemOp>(sourceOp))
      name = alloc.getUniqName();
    else
      unknownAllocOp = true;

    if (auto declOp = source.origin.instantiationPoint) {
      // Use the outermost scope for local allocations,
      // because using the innermost scope may result
      // in incorrect TBAA, when calls are inlined in MLIR.
      auto declareOp = mlir::dyn_cast<fir::DeclareOp>(declOp);
      assert(declareOp && "Instantiation point must be fir.declare");
      scopeOp = state.getOutermostScope(declareOp);
    }

    if (unknownAllocOp) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "WARN: unknown defining op for SourceKind::Allocate " << *op
                 << "\n");
    } else if (source.isPointer() && state.attachLocalAllocTag()) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Found reference to allocation at " << *op << "\n");
      tag = state.getFuncTreeWithScope(func, scopeOp).targetDataTree.getTag();
    } else if (name && state.attachLocalAllocTag()) {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Found reference to allocation "
                                        << name << " at " << *op << "\n");
      tag = state.getFuncTreeWithScope(func, scopeOp)
                .allocatedDataTree.getTag(*name);
    } else if (state.attachLocalAllocTag()) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "WARN: couldn't find a name for allocation " << *op
                 << "\n");
      tag =
          state.getFuncTreeWithScope(func, scopeOp).allocatedDataTree.getTag();
    }
  } else {
    if (source.kind != fir::AliasAnalysis::SourceKind::Argument &&
        source.kind != fir::AliasAnalysis::SourceKind::Allocate &&
        source.kind != fir::AliasAnalysis::SourceKind::Global)
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "WARN: unsupported value: " << source << "\n");
  }

  if (tag)
    op.setTBAATags(mlir::ArrayAttr::get(&getContext(), tag));
}

void AddAliasTagsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");

  // MLIR forbids storing state in a pass because different instances might be
  // used in different threads.
  // Instead this pass stores state per mlir::ModuleOp (which is what MLIR
  // thinks the pass operates on), then the real work of the pass is done in
  // runOnAliasInterface
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  mlir::ModuleOp module = getOperation();
  mlir::DataLayout dl = *fir::support::getOrSetMLIRDataLayout(
      module, /*allowDefaultLayout=*/false);
  PassState state(module, dl, domInfo,
                  localAllocsThreshold.getPosition()
                      ? std::optional<unsigned>(localAllocsThreshold)
                      : std::nullopt);

  module.walk(
      [&](fir::FirAliasTagOpInterface op) { runOnAliasInterface(op, state); });

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}
