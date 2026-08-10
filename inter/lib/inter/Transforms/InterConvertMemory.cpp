// inter-convert-memory: llvm.load/store -> xw.load/store with explicit
// token edges from alias analysis (design doc section 9).
//
// v1 AA: distinct kernel pointer args cannot alias (OpenCL restrict
// semantics); SLM (addrspace 3) never aliases global; same-class hazards are
// ordered. A barrier orders against every live class chain via a join.

#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace inter {
#define GEN_PASS_DEF_CONVERTMEMORY
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

struct ConvertMemory : public inter::impl::ConvertMemoryBase<ConvertMemory> {
  static constexpr int kUnknownAliasClass = -1;
  static constexpr int kSlmAliasClass = 1000;

  struct AliasState {
    Value lastWrite;
    SmallVector<Value, 2> reads;
  };

  struct MemoryState {
    llvm::MapVector<int, AliasState> aliases;
    Value floor;
  };

  Type tokenType;

  // Alias class keys are kernel argument indices or one class for all SLM
  // globals. Unknown addresses alias every class.
  int aliasClass(Value address) const {
    while (LLVM::GEPOp gep = address.getDefiningOp<LLVM::GEPOp>())
      address = gep.getBase();

    if (BlockArgument argument = dyn_cast<BlockArgument>(address)) {
      Block *owner = argument.getOwner();
      if (owner->isEntryBlock() && isa<func::FuncOp>(owner->getParentOp()))
        return argument.getArgNumber();
    }

    if (address.getDefiningOp<LLVM::AddressOfOp>()) {
      LLVM::LLVMPointerType pointerType =
          dyn_cast<LLVM::LLVMPointerType>(address.getType());
      if (pointerType && pointerType.getAddressSpace() == 3)
        return kSlmAliasClass;
    }
    return kUnknownAliasClass;
  }

  Value joinDependencies(OpBuilder &builder, Location loc, Value entry,
                         ArrayRef<Value> dependencies) const {
    if (dependencies.empty())
      return entry;
    if (dependencies.size() == 1)
      return dependencies.front();
    return xw::TokenJoinOp::create(builder, loc, entry.getType(), dependencies)
        .getToken();
  }

  SmallVector<Value> collectDependencies(const MemoryState &state, int cls,
                                         bool includeReads,
                                         Value existing = Value()) const {
    SmallVector<Value> dependencies;
    llvm::SmallPtrSet<Value, 8> seen;
    auto append = [&](Value token) {
      if (token && seen.insert(token).second)
        dependencies.push_back(token);
    };
    auto appendState = [&](const AliasState &alias) {
      if (includeReads && !alias.reads.empty()) {
        for (Value read : alias.reads)
          append(read);
        return;
      }
      append(alias.lastWrite);
    };

    append(existing);
    if (cls == kUnknownAliasClass) {
      for (const auto &item : state.aliases)
        appendState(item.second);
    } else {
      auto alias = state.aliases.find(cls);
      if (alias != state.aliases.end())
        appendState(alias->second);
      auto unknown = state.aliases.find(kUnknownAliasClass);
      if (unknown != state.aliases.end())
        appendState(unknown->second);
    }
    if (dependencies.empty() || existing)
      append(state.floor);
    return dependencies;
  }

  Value closeState(OpBuilder &builder, Location loc,
                   const MemoryState &state) const {
    SmallVector<Value> dependencies;
    llvm::SmallPtrSet<Value, 8> seen;
    auto append = [&](Value token) {
      if (token && seen.insert(token).second)
        dependencies.push_back(token);
    };
    for (const auto &item : state.aliases) {
      const AliasState &alias = item.second;
      if (!alias.reads.empty()) {
        for (Value read : alias.reads)
          append(read);
      } else {
        append(alias.lastWrite);
      }
    }
    return joinDependencies(builder, loc, state.floor, dependencies);
  }

  RegionBranchOpInterface appendTokenResult(RegionBranchOpInterface branch,
                                            Value &token) const {
    Operation *operation = branch.getOperation();
    unsigned oldResultCount = operation->getNumResults();
    SmallVector<Type> resultTypes(operation->getResultTypes());
    resultTypes.push_back(tokenType);

    Operation *replacement = Operation::create(
        operation->getLoc(), operation->getName(), resultTypes,
        operation->getOperands(), operation->getAttrDictionary(),
        operation->getPropertiesStorage(), operation->getSuccessors(),
        operation->getNumRegions());
    for (auto [oldRegion, newRegion] :
         llvm::zip(operation->getRegions(), replacement->getRegions()))
      newRegion.takeBody(oldRegion);

    OpBuilder(operation).insert(replacement);
    operation->replaceAllUsesWith(
        replacement->getResults().take_front(oldResultCount));
    operation->erase();
    token = replacement->getResult(oldResultCount);
    return cast<RegionBranchOpInterface>(replacement);
  }

  LogicalResult threadRegionBranch(RegionBranchOpInterface branch,
                                   Value incoming, Value &outgoing) {
    // RegionBranchOpInterface describes edges but cannot grow their inputs.
    // Follow MLIR's append-at-tail convention and validate every changed edge.
    Operation *operation = branch.getOperation();
    unsigned regionCount = operation->getNumRegions();
    SmallVector<bool> threadedRegions(regionCount, false);

    for (Region &region : operation->getRegions()) {
      if (!region.hasOneBlock())
        return operation->emitOpError(
                   "requires single-block regions for memory-token threading"),
               failure();
      if (!isa<RegionBranchTerminatorOpInterface>(
              region.front().getTerminator()))
        return operation->emitOpError("region terminator does not implement "
                                      "RegionBranchTerminatorOpInterface"),
               failure();

      SmallVector<RegionBranchPoint> predecessors;
      branch.getPredecessors(RegionSuccessor(&region), predecessors);
      threadedRegions[region.getRegionNumber()] =
          llvm::any_of(predecessors, [](RegionBranchPoint point) {
            return !point.isParent();
          });
    }

    SmallVector<RegionSuccessor> parentSuccessors;
    branch.getSuccessorRegions(RegionBranchPoint::parent(), parentSuccessors);
    bool parentExits =
        llvm::any_of(parentSuccessors, [](RegionSuccessor successor) {
          return successor.isOperation();
        });
    bool hasThreadedRegion = llvm::is_contained(threadedRegions, true);
    if (parentExits && !hasThreadedRegion)
      return operation->emitOpError(
                 "requires an explicit region for every control-flow path"),
             failure();

    if (hasThreadedRegion)
      operation->insertOperands(operation->getNumOperands(), incoming);
    for (Region &region : operation->getRegions()) {
      if (threadedRegions[region.getRegionNumber()])
        region.front().addArgument(tokenType, operation->getLoc());
    }

    branch = appendTokenResult(branch, outgoing);
    operation = branch.getOperation();

    ValueRange resultInputs =
        branch.getSuccessorInputs(RegionSuccessor(operation));
    if (resultInputs.empty() || resultInputs.back() != outgoing)
      return operation->emitOpError(
                 "must expose appended results as successor inputs"),
             failure();

    for (Region &region : operation->getRegions()) {
      unsigned regionNumber = region.getRegionNumber();
      if (!threadedRegions[regionNumber])
        continue;
      Value tokenArgument = region.front().getArguments().back();
      ValueRange inputs = branch.getSuccessorInputs(RegionSuccessor(&region));
      if (inputs.empty() || inputs.back() != tokenArgument)
        return operation->emitOpError(
                   "must expose appended block arguments as successor inputs"),
               failure();
    }

    parentSuccessors.clear();
    branch.getSuccessorRegions(RegionBranchPoint::parent(), parentSuccessors);
    for (RegionSuccessor successor : parentSuccessors) {
      Value target;
      if (successor.isOperation()) {
        target = outgoing;
      } else {
        Region *region = successor.getSuccessor();
        if (!threadedRegions[region->getRegionNumber()])
          continue;
        target = region->front().getArguments().back();
      }
      OperandRange operands = branch.getEntrySuccessorOperands(successor);
      ValueRange inputs = branch.getSuccessorInputs(successor);
      if (operands.size() != inputs.size() || operands.empty() ||
          operands.back() != incoming || inputs.back() != target)
        return operation->emitOpError(
                   "does not expose an appendable entry successor operand"),
               failure();
    }

    SmallVector<Value> regionOutputs(regionCount);
    for (Region &region : operation->getRegions()) {
      unsigned regionNumber = region.getRegionNumber();
      Value regionInput = threadedRegions[regionNumber]
                              ? region.front().getArguments().back()
                              : incoming;
      if (failed(convertBlock(region.front(), regionInput,
                              regionOutputs[regionNumber])))
        return failure();
    }

    for (Region &region : operation->getRegions()) {
      unsigned regionNumber = region.getRegionNumber();
      auto terminator = cast<RegionBranchTerminatorOpInterface>(
          region.front().getTerminator());
      RegionBranchPoint point(terminator);
      SmallVector<RegionSuccessor> successors;
      branch.getSuccessorRegions(point, successors);
      for (RegionSuccessor successor : successors) {
        Value target;
        if (successor.isOperation()) {
          target = outgoing;
        } else {
          Region *successorRegion = successor.getSuccessor();
          if (!threadedRegions[successorRegion->getRegionNumber()])
            continue;
          target = successorRegion->front().getArguments().back();
        }

        OperandRange operands = branch.getSuccessorOperands(point, successor);
        ValueRange inputs = branch.getSuccessorInputs(successor);
        if (inputs.empty() || inputs.back() != target)
          return operation->emitOpError(
                     "does not expose the threaded token as a successor input"),
                 failure();
        if (operands.size() + 1 == inputs.size()) {
          terminator.getMutableSuccessorOperands(successor).append(
              regionOutputs[regionNumber]);
          operands = branch.getSuccessorOperands(point, successor);
        }
        if (operands.size() != inputs.size() || operands.empty() ||
            operands.back() != regionOutputs[regionNumber])
          return operation->emitOpError(
                     "does not expose an appendable terminator successor "
                     "operand"),
                 failure();
      }
    }
    return success();
  }

  void runOnOperation() override {
    func::FuncOp kernel = getOperation();
    if (!kernel->hasAttr("xemachine.kernel"))
      return;
    if (failed(convertKernel(kernel)))
      return signalPassFailure();
  }

  LogicalResult convertKernel(func::FuncOp kernel) {
    if (!kernel.getBody().hasOneBlock())
      return kernel.emitOpError(
                 "requires structured control flow in a single function block"),
             failure();

    tokenType = inter::xemachine::MemTokenType::get(kernel.getContext());
    Block &entryBlock = kernel.getBody().front();
    OpBuilder builder = OpBuilder::atBlockBegin(&entryBlock);
    Value entry =
        xw::TokenOp::create(builder, kernel.getLoc(), tokenType).getToken();
    Value outgoing;
    return convertBlock(entryBlock, entry, outgoing);
  }

  LogicalResult convertBlock(Block &block, Value incoming, Value &outgoing) {
    MemoryState state;
    state.floor = incoming;
    SmallVector<Operation *> operations;
    for (Operation &operation : block)
      operations.push_back(&operation);

    for (Operation *operation : operations) {
      if (auto load = dyn_cast<LLVM::LoadOp>(operation)) {
        OpBuilder builder(operation);
        int cls = aliasClass(load.getAddr());
        SmallVector<Value> dependencies =
            collectDependencies(state, cls, /*includeReads=*/false);
        Value dependency = joinDependencies(builder, operation->getLoc(),
                                            state.floor, dependencies);
        xw::LoadOp converted =
            xw::LoadOp::create(builder, load.getLoc(), load.getType(),
                               tokenType, load.getAddr(), dependency);
        load->replaceAllUsesWith(ValueRange{converted.getValue()});
        state.aliases[cls].reads.push_back(converted.getToken());
        load->erase();
      } else if (auto store = dyn_cast<LLVM::StoreOp>(operation)) {
        OpBuilder builder(operation);
        int cls = aliasClass(store.getAddr());
        SmallVector<Value> dependencies =
            collectDependencies(state, cls, /*includeReads=*/true);
        Value dependency = joinDependencies(builder, operation->getLoc(),
                                            state.floor, dependencies);
        xw::StoreOp converted =
            xw::StoreOp::create(builder, store.getLoc(), tokenType,
                                store.getAddr(), store.getValue(), dependency);
        if (cls == kUnknownAliasClass)
          state.aliases.clear();
        AliasState &alias = state.aliases[cls];
        alias.lastWrite = converted.getToken();
        alias.reads.clear();
        store->erase();
      } else if (auto atomic = dyn_cast<xw::AtomicAddOp>(operation)) {
        OpBuilder builder(operation);
        int cls = aliasClass(atomic.getAddress());
        SmallVector<Value> dependencies = collectDependencies(
            state, cls, /*includeReads=*/true, atomic.getDependency());
        Value dependency = joinDependencies(builder, operation->getLoc(),
                                            state.floor, dependencies);
        xw::AtomicAddOp converted = xw::AtomicAddOp::create(
            builder, operation->getLoc(), atomic.getOld().getType(), tokenType,
            atomic.getAddress(), atomic.getValue(), dependency);
        atomic.getOld().replaceAllUsesWith(converted.getOld());
        atomic.getToken().replaceAllUsesWith(converted.getToken());
        atomic->erase();
        if (cls == kUnknownAliasClass)
          state.aliases.clear();
        AliasState &alias = state.aliases[cls];
        alias.lastWrite = converted.getToken();
        alias.reads.clear();
      } else if (auto barrier = dyn_cast<xw::BarrierOp>(operation)) {
        OpBuilder builder(operation);
        SmallVector<Value> dependencies =
            collectDependencies(state, kUnknownAliasClass,
                                /*includeReads=*/true, barrier.getDependency());
        Value dependency = joinDependencies(builder, operation->getLoc(),
                                            state.floor, dependencies);
        xw::BarrierOp converted = xw::BarrierOp::create(
            builder, operation->getLoc(), tokenType, dependency);
        barrier.getToken().replaceAllUsesWith(converted.getToken());
        barrier->erase();
        state.aliases.clear();
        state.floor = converted.getToken();
      } else if (auto branch = dyn_cast<RegionBranchOpInterface>(operation)) {
        OpBuilder builder(operation);
        Value branchInput = closeState(builder, operation->getLoc(), state);
        Value branchOutput;
        if (failed(threadRegionBranch(branch, branchInput, branchOutput)))
          return failure();
        state.aliases.clear();
        state.floor = branchOutput;
      } else if (operation->getNumRegions() != 0) {
        return operation->emitOpError(
                   "with regions must implement RegionBranchOpInterface"),
               failure();
      }
    }

    Operation *terminator = block.getTerminator();
    if (!terminator)
      return emitError(block.getParent()->getLoc(),
                       "memory-token threading requires a terminator"),
             failure();
    OpBuilder builder(terminator);
    outgoing = closeState(builder, terminator->getLoc(), state);
    return success();
  }
};

} // namespace
