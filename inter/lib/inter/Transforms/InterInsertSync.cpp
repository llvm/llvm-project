// Assign Xe scoreboard tokens and materialize precise waits after scheduling
// and physical register allocation. Token pseudos remain zero-byte bookkeeping.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <limits>
#include <optional>

namespace inter {
#define GEN_PASS_DEF_INSERTSYNC
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace mlir::dataflow;
using namespace inter::xemachine;

namespace {

struct RegisterSpan {
  int64_t begin;
  int64_t end;

  bool operator==(const RegisterSpan &other) const {
    return begin == other.begin && end == other.end;
  }

  bool operator!=(const RegisterSpan &other) const { return !(*this == other); }

  bool overlaps(const RegisterSpan &other) const {
    return begin < other.end && other.begin < end;
  }
};

struct IssueTicket {
  Operation *issue;
  SmallVector<RegisterSpan, 4> sources;
  SmallVector<RegisterSpan, 2> destinations;
  bool sourcePending = false;
  bool destinationPending = false;
};

struct ValueTicket {
  Value id;
  Operation *issue;
  bool sourcePending = false;
  bool destinationPending = false;
};

struct SyncState {
  SmallVector<IssueTicket, 8> issues;
  SmallVector<ValueTicket, 8> values;
  bool initialized = false;
};

static bool insertSpan(SmallVectorImpl<RegisterSpan> &spans,
                       RegisterSpan span) {
  if (llvm::is_contained(spans, span))
    return false;
  spans.push_back(span);
  return true;
}

static bool insertIssue(SyncState &state, IssueTicket ticket) {
  for (IssueTicket &existing : state.issues) {
    if (existing.issue != ticket.issue)
      continue;
    bool changed = false;
    for (RegisterSpan span : ticket.sources)
      changed |= insertSpan(existing.sources, span);
    for (RegisterSpan span : ticket.destinations)
      changed |= insertSpan(existing.destinations, span);
    if (ticket.sourcePending && !existing.sourcePending) {
      existing.sourcePending = true;
      changed = true;
    }
    if (ticket.destinationPending && !existing.destinationPending) {
      existing.destinationPending = true;
      changed = true;
    }
    return changed;
  }
  state.issues.push_back(std::move(ticket));
  return true;
}

static bool insertValue(SyncState &state, ValueTicket ticket) {
  for (ValueTicket &existing : state.values) {
    if (existing.id != ticket.id || existing.issue != ticket.issue)
      continue;
    bool changed = false;
    if (ticket.sourcePending && !existing.sourcePending) {
      existing.sourcePending = true;
      changed = true;
    }
    if (ticket.destinationPending && !existing.destinationPending) {
      existing.destinationPending = true;
      changed = true;
    }
    return changed;
  }
  state.values.push_back(ticket);
  return true;
}

class SyncLattice : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SyncLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  const SyncState &get() const { return state; }

  ChangeResult joinWith(const SyncState &incoming) {
    if (!incoming.initialized)
      return ChangeResult::NoChange;
    if (!state.initialized) {
      state = incoming;
      return ChangeResult::Change;
    }
    bool changed = false;
    for (const IssueTicket &ticket : incoming.issues)
      changed |= insertIssue(state, ticket);
    for (const ValueTicket &ticket : incoming.values)
      changed |= insertValue(state, ticket);
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    return joinWith(static_cast<const SyncLattice &>(rhs).state);
  }

  ChangeResult setEntryState() {
    SyncState entry;
    entry.initialized = true;
    if (state.initialized && state.issues.empty() && state.values.empty())
      return ChangeResult::NoChange;
    state = entry;
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const override {
    os << "initialized=" << state.initialized
       << " issues=" << state.issues.size()
       << " values=" << state.values.size();
  }

private:
  SyncState state;
};

struct TokenWait {
  unsigned sbid;
  SWSBTokenMode mode;

  bool operator==(const TokenWait &other) const {
    return sbid == other.sbid && mode == other.mode;
  }
};

struct IssueWait {
  Operation *issue;
  SWSBTokenMode mode;

  bool operator==(const IssueWait &other) const {
    return issue == other.issue && mode == other.mode;
  }
};

using AllocationPlan = DenseMap<Operation *, unsigned>;

static bool isDpasChainPredecessor(Operation *candidate, DpasOp consumer);

static std::optional<RegisterSpan> getRegisterSpan(Value value) {
  if (RegType type = dyn_cast<RegType>(value.getType())) {
    if (type.getBaseGRF() < 0 || type.getWidthDwords() == 0)
      return std::nullopt;
    int64_t begin = type.getBaseGRF() * 64;
    int64_t widthDwords = type.getWidthDwords();
    for (OpOperand &use : value.getUses()) {
      UpdateTupleOp update = dyn_cast<UpdateTupleOp>(use.getOwner());
      if (!update || use.getOperandNumber() == 0)
        continue;
      RegType baseType = cast<RegType>(update.getBase().getType());
      unsigned updateIndex = use.getOperandNumber() - 1;
      int64_t offset =
          cast<IntegerAttr>(update.getOffsets()[updateIndex]).getInt();
      assert(offset >= 0 && offset <= baseType.getWidthDwords() &&
             "verified tuple update offset must fit its base storage");
      widthDwords =
          std::max<int64_t>(widthDwords, baseType.getWidthDwords() - offset);
    }
    return RegisterSpan{begin, begin + widthDwords * 4};
  }
  if (ARFType type = dyn_cast<ARFType>(value.getType())) {
    if (type.getIndex() < 0 || type.getWidthDwords() == 0)
      return std::nullopt;
    constexpr int64_t arfBase = int64_t{1} << 32;
    int64_t begin = arfBase + static_cast<int64_t>(type.getFile()) * (1 << 20) +
                    type.getIndex() * 64;
    return RegisterSpan{begin, begin + type.getWidthDwords() * 4};
  }
  return std::nullopt;
}

static Block *getDefiningBlock(Value value) {
  if (Operation *operation = value.getDefiningOp())
    return operation->getBlock();
  if (BlockArgument argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner();
  return nullptr;
}

// Values that do not dominate an edge target lose their SSA alias. The issue
// itself remains live because its physical scoreboard obligation survives.
static void collapseEscaping(SyncState &state, Block *target,
                             DominanceInfo &dominance) {
  llvm::erase_if(state.values, [&](const ValueTicket &ticket) {
    Block *definition = getDefiningBlock(ticket.id);
    return definition && !dominance.dominates(definition, target);
  });
}

static void applyReadWait(SyncState &state) {
  for (IssueTicket &ticket : state.issues)
    ticket.sourcePending = false;
  for (ValueTicket &ticket : state.values)
    ticket.sourcePending = false;
  llvm::erase_if(state.issues, [](const IssueTicket &ticket) {
    return !ticket.sourcePending && !ticket.destinationPending;
  });
  llvm::erase_if(state.values, [](const ValueTicket &ticket) {
    return !ticket.sourcePending && !ticket.destinationPending;
  });
}

static void applyWriteWait(SyncState &state) {
  llvm::SmallDenseSet<Operation *> completed;
  for (const IssueTicket &ticket : state.issues)
    if (ticket.destinationPending)
      completed.insert(ticket.issue);
  llvm::erase_if(state.issues, [&](const IssueTicket &ticket) {
    return completed.contains(ticket.issue);
  });
  llvm::erase_if(state.values, [&](const ValueTicket &ticket) {
    return completed.contains(ticket.issue);
  });
}

static void applyIssueWait(SyncState &state, IssueWait wait) {
  for (IssueTicket &ticket : state.issues) {
    if (ticket.issue != wait.issue)
      continue;
    if (wait.mode == SWSBTokenMode::source)
      ticket.sourcePending = false;
    else {
      assert(wait.mode == SWSBTokenMode::destination &&
             "token waits must name a completion mode");
      ticket.sourcePending = false;
      ticket.destinationPending = false;
    }
  }
  llvm::erase_if(state.issues, [](const IssueTicket &ticket) {
    return !ticket.sourcePending && !ticket.destinationPending;
  });
  for (ValueTicket &ticket : state.values) {
    IssueTicket *issue =
        llvm::find_if(state.issues, [&](const IssueTicket &it) {
          return it.issue == ticket.issue;
        });
    if (issue == state.issues.end()) {
      ticket.sourcePending = false;
      ticket.destinationPending = false;
      continue;
    }
    ticket.sourcePending &= issue->sourcePending;
    ticket.destinationPending &= issue->destinationPending;
  }
  llvm::erase_if(state.values, [](const ValueTicket &ticket) {
    return !ticket.sourcePending && !ticket.destinationPending;
  });
}

static void applyTokenWait(SyncState &state, TokenWait wait) {
  SmallVector<Operation *> matchingIssues;
  for (const IssueTicket &ticket : state.issues) {
    FinalSWSB swsb = cast<SWSBInfoOpInterface>(ticket.issue).getFinalSWSB();
    if (swsb.token == static_cast<int32_t>(wait.sbid))
      matchingIssues.push_back(ticket.issue);
  }
  for (Operation *issue : matchingIssues)
    applyIssueWait(state, IssueWait{issue, wait.mode});
}

static void applyWriteWait(SyncState &state, uint32_t sbidMask) {
  for (unsigned sbid : llvm::seq<unsigned>(0, 32))
    if (sbidMask & (uint32_t{1} << sbid))
      applyTokenWait(state, TokenWait{sbid, SWSBTokenMode::destination});
}

static void requireWait(SmallVectorImpl<IssueWait> &requirements,
                        IssueWait wait) {
  if (wait.mode == SWSBTokenMode::source &&
      llvm::is_contained(requirements,
                         IssueWait{wait.issue, SWSBTokenMode::destination}))
    return;
  if (wait.mode == SWSBTokenMode::destination)
    llvm::erase(requirements, IssueWait{wait.issue, SWSBTokenMode::source});
  if (!llvm::is_contained(requirements, wait))
    requirements.push_back(wait);
}

static void requireWait(SmallVectorImpl<IssueWait> &requirements,
                        const IssueTicket &ticket, SWSBTokenMode mode) {
  requireWait(requirements, IssueWait{ticket.issue, mode});
}

static void requireValue(SmallVectorImpl<IssueWait> &requirements, Value value,
                         const SyncState &state) {
  std::optional<RegisterSpan> span = getRegisterSpan(value);
  for (const ValueTicket &valueTicket : state.values) {
    if (valueTicket.id != value)
      continue;
    auto issue = llvm::find_if(state.issues, [&](const IssueTicket &ticket) {
      return ticket.issue == valueTicket.issue;
    });
    if (issue == state.issues.end())
      continue;
    if (valueTicket.destinationPending)
      requireWait(requirements, *issue, SWSBTokenMode::destination);
    else if (valueTicket.sourcePending)
      requireWait(requirements, *issue, SWSBTokenMode::source);
  }
  if (!span)
    return;
  for (const IssueTicket &ticket : state.issues)
    if (ticket.destinationPending &&
        llvm::any_of(ticket.destinations, [&](RegisterSpan destination) {
          return span->overlaps(destination);
        }))
      requireWait(requirements, ticket, SWSBTokenMode::destination);
}

static void requireDefinition(SmallVectorImpl<IssueWait> &requirements,
                              RegisterSpan definition, const SyncState &state,
                              DpasOp chainedConsumer = nullptr) {
  for (const IssueTicket &ticket : state.issues) {
    if (chainedConsumer &&
        isDpasChainPredecessor(ticket.issue, chainedConsumer))
      continue;
    if (ticket.sourcePending &&
        llvm::any_of(ticket.sources, [&](RegisterSpan source) {
          return definition.overlaps(source);
        }))
      requireWait(requirements, ticket, SWSBTokenMode::source);
    if (ticket.destinationPending &&
        llvm::any_of(ticket.destinations, [&](RegisterSpan destination) {
          return definition.overlaps(destination);
        }))
      requireWait(requirements, ticket, SWSBTokenMode::destination);
  }
}

static bool isForwardedRegionOperand(OpOperand &operand) {
  Operation *operation = operand.getOwner();
  if (RegionBranchOpInterface branch =
          dyn_cast<RegionBranchOpInterface>(operation)) {
    RegionBranchSuccessorMapping mapping;
    branch.getSuccessorOperandInputMapping(mapping,
                                           RegionBranchPoint::parent());
    return mapping.contains(&operand);
  }

  RegionBranchTerminatorOpInterface terminator =
      dyn_cast<RegionBranchTerminatorOpInterface>(operation);
  if (!terminator)
    return false;
  RegionBranchOpInterface branch =
      dyn_cast<RegionBranchOpInterface>(operation->getParentOp());
  if (!branch)
    return false;
  RegionBranchSuccessorMapping mapping;
  branch.getSuccessorOperandInputMapping(mapping,
                                         RegionBranchPoint(terminator));
  return mapping.contains(&operand);
}

static bool isForwardedBlockOperand(OpOperand &operand) {
  BranchOpInterface branch = dyn_cast<BranchOpInterface>(operand.getOwner());
  if (!branch)
    return false;
  for (unsigned index = 0, end = operand.getOwner()->getNumSuccessors();
       index < end; ++index) {
    MutableOperandRange forwarded =
        branch.getSuccessorOperands(index).getMutableForwardedOperands();
    for (OpOperand &candidate : forwarded)
      if (&candidate == &operand)
        return true;
  }
  return false;
}

static bool isForwardedControlOperand(OpOperand &operand) {
  return isForwardedRegionOperand(operand) || isForwardedBlockOperand(operand);
}

static bool isControlFlowOp(Operation *operation) {
  return isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
             BranchOpInterface>(operation);
}

static bool emitsMachineInstruction(Operation *operation) {
  return !operation->hasTrait<OpTrait::xemachine::NoMachineInst>();
}

static bool isFullDrain(Operation *operation) {
  return operation->hasTrait<OpTrait::xemachine::FullScoreboardDrain>() ||
         operation->hasAttr("eot");
}

static bool isDpasChainPredecessor(Operation *candidate, DpasOp consumer) {
  Operation *producer = consumer.getAcc().getDefiningOp();
  while (DpasOp dpas = dyn_cast_or_null<DpasOp>(producer)) {
    if (producer == candidate)
      return true;
    producer = dpas.getAcc().getDefiningOp();
  }
  return false;
}

static SmallVector<IssueWait> computeRequirement(Operation *operation,
                                                 const SyncState &state) {
  SmallVector<IssueWait> requirements;
  if (isa<PayloadPrologueEndOp>(operation)) {
    for (const IssueTicket &ticket : state.issues)
      requireWait(requirements, ticket,
                  ticket.destinationPending ? SWSBTokenMode::destination
                                            : SWSBTokenMode::source);
    return requirements;
  }
  if (isFullDrain(operation)) {
    for (const IssueTicket &ticket : state.issues)
      requireWait(requirements, ticket,
                  ticket.destinationPending ? SWSBTokenMode::destination
                                            : SWSBTokenMode::source);
    return requirements;
  }

  if (!emitsMachineInstruction(operation))
    return requirements;

  for (OpOperand &operand : operation->getOpOperands()) {
    if (isControlFlowOp(operation) && isForwardedControlOperand(operand))
      continue;
    if (DpasOp dpas = dyn_cast<DpasOp>(operation);
        dpas && operand.get() == dpas.getAcc() &&
        isa_and_nonnull<DpasOp>(operand.get().getDefiningOp()))
      continue;
    requireValue(requirements, operand.get(), state);
  }

  if (isa<RegionBranchOpInterface>(operation))
    return requirements;
  DpasOp dpas = dyn_cast<DpasOp>(operation);
  for (Value result : operation->getResults())
    if (std::optional<RegisterSpan> span = getRegisterSpan(result))
      requireDefinition(requirements, *span, state, dpas);
  return requirements;
}

static void deriveValue(SyncState &state, ValueRange sources,
                        Value destination) {
  SmallVector<ValueTicket> derived;
  for (Value source : sources) {
    for (const ValueTicket &ticket : state.values) {
      if (ticket.id != source)
        continue;
      ValueTicket alias = ticket;
      alias.id = destination;
      derived.push_back(alias);
    }
  }
  for (ValueTicket ticket : derived)
    insertValue(state, ticket);
}

static void deriveValue(SyncState &state, Value source, Value destination) {
  deriveValue(state, ValueRange(source), destination);
}

static void deriveResults(Operation *operation, SyncState &state) {
  for (Value result : operation->getResults())
    deriveValue(state, operation->getOperands(), result);
}

static void recordIssue(AsyncScoreboardOpInterface operation,
                        SyncState &state) {
  IssueTicket issue{operation.getOperation()};
  issue.sourcePending = true;
  issue.destinationPending = operation.hasAsyncDestination();
  for (Value operand : operation->getOperands())
    if (std::optional<RegisterSpan> span = getRegisterSpan(operand))
      insertSpan(issue.sources, *span);
  for (Value result : operation->getResults()) {
    if (std::optional<RegisterSpan> span = getRegisterSpan(result))
      insertSpan(issue.destinations, *span);
  }
  insertIssue(state, issue);
  for (Value result : operation->getResults()) {
    bool token = isa<MemTokenType>(result.getType());
    bool destination = !token || issue.destinationPending;
    insertValue(state, ValueTicket{result, issue.issue, token && !destination,
                                   destination});
  }
}

template <typename EmitFn>
static void applyDrain(Operation *operation, SyncState &state, EmitFn emit) {
  SmallVector<IssueWait> requirements = computeRequirement(operation, state);
  emit(operation, requirements);
  for (IssueWait wait : requirements)
    applyIssueWait(state, wait);
}

static void observeSync(SyncOp sync, SyncState &state) {
  if (sync.getKind() == SyncKind::nop) {
    FinalSWSB swsb = sync.getFinalSWSB();
    if (swsb.token >= 0 && swsb.tokenMode != SWSBTokenMode::set)
      applyTokenWait(
          state, TokenWait{static_cast<unsigned>(swsb.token), swsb.tokenMode});
  } else if (sync.getKind() == SyncKind::allrd)
    applyReadWait(state);
  else if (sync.getKind() == SyncKind::allwr) {
    uint32_t sbidMask = sync.getSbidMask();
    if (sbidMask != 0)
      applyWriteWait(state, sbidMask);
    else
      applyWriteWait(state);
  } else if (sync.getKind() == SyncKind::bar)
    applyReadWait(state);
  deriveResults(sync, state);
}

template <typename EmitFn>
static void runTransfer(Operation *operation, SyncState &state, EmitFn emit) {
  if (SyncOp sync = dyn_cast<SyncOp>(operation)) {
    observeSync(sync, state);
    return;
  }

  if (isa<PayloadPrologueOp>(operation))
    return;

  if (isa<RegionBranchOpInterface>(operation)) {
    applyDrain(operation, state, emit);
    return;
  }

  if (isa<RegionBranchTerminatorOpInterface, BranchOpInterface>(operation)) {
    applyDrain(operation, state, emit);
    return;
  }

  if (!emitsMachineInstruction(operation)) {
    if (!operation->hasTrait<OpTrait::xemachine::CompletionFree>())
      deriveResults(operation, state);
    return;
  }

  applyDrain(operation, state, emit);
  if (auto async = dyn_cast<AsyncScoreboardOpInterface>(operation);
      async && !isFullDrain(operation))
    recordIssue(async, state);
  else
    deriveResults(operation, state);
}

static void propagateValues(ValueRange sources, ValueRange destinations,
                            SyncState &state) {
  for (auto [source, destination] : llvm::zip_equal(sources, destinations))
    deriveValue(state, source, destination);
}

static void propagateBranchOperands(Operation *terminator, Block *successor,
                                    SyncState &state) {
  BranchOpInterface branch = dyn_cast<BranchOpInterface>(terminator);
  if (!branch)
    return;
  for (auto [index, target] : llvm::enumerate(terminator->getSuccessors())) {
    if (target != successor)
      continue;
    SuccessorOperands operands = branch.getSuccessorOperands(index);
    unsigned limit =
        std::min<unsigned>(operands.size(), successor->getNumArguments());
    for (unsigned argument = 0; argument < limit; ++argument)
      if (Value source = operands[argument])
        deriveValue(state, source, successor->getArgument(argument));
  }
}

static bool hasSuccessor(RegionBranchOpInterface branch,
                         RegionBranchPoint point, RegionSuccessor expected) {
  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(point, successors);
  return llvm::is_contained(successors, expected);
}

static void propagateRegionOperands(RegionBranchOpInterface branch,
                                    std::optional<unsigned> regionFrom,
                                    RegionSuccessor successor,
                                    SyncState &state) {
  ValueRange destinations = branch.getSuccessorInputs(successor);
  if (!regionFrom) {
    propagateValues(branch.getEntrySuccessorOperands(successor), destinations,
                    state);
    return;
  }

  Region &region = branch->getRegion(*regionFrom);
  for (Block &block : region) {
    RegionBranchTerminatorOpInterface terminator =
        dyn_cast_or_null<RegionBranchTerminatorOpInterface>(
            block.getTerminator());
    if (!terminator ||
        !hasSuccessor(branch, RegionBranchPoint(terminator), successor))
      continue;
    propagateValues(terminator.getSuccessorOperands(successor), destinations,
                    state);
  }
}

class SyncAnalysis : public DenseForwardDataFlowAnalysis<SyncLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SyncAnalysis)

  SyncAnalysis(DataFlowSolver &solver, DominanceInfo &dominance)
      : DenseForwardDataFlowAnalysis(solver), dominance(dominance) {}

  LogicalResult initialize(Operation *top) override {
    auto markRegions = [&](Operation *operation) {
      for (Region &region : operation->getRegions()) {
        for (Block &block : region) {
          Executable *blockLive =
              getOrCreate<Executable>(getProgramPointBefore(&block));
          propagateIfChanged(blockLive, blockLive->setToLive());
          Operation *terminator = block.getTerminator();
          if (!terminator)
            continue;
          for (Block *successor : terminator->getSuccessors()) {
            Executable *edgeLive = getOrCreate<Executable>(
                getLatticeAnchor<CFGEdge>(&block, successor));
            propagateIfChanged(edgeLive, edgeLive->setToLive());
          }
        }
      }
    };
    markRegions(top);
    top->walk(markRegions);
    return DenseForwardDataFlowAnalysis<SyncLattice>::initialize(top);
  }

  void setToEntryState(SyncLattice *lattice) override {
    propagateIfChanged(lattice, lattice->setEntryState());
  }

  LogicalResult visitOperation(Operation *operation, const SyncLattice &before,
                               SyncLattice *after) override {
    if (!before.get().initialized)
      return success();
    SyncState next = before.get();
    transfer(operation, next);
    propagateIfChanged(after, after->joinWith(next));
    markCFGSuccessorsLive(operation, next);
    return success();
  }

  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const SyncLattice &before,
                          SyncLattice *after) override {
    SyncState next = before.get();
    propagateBranchOperands(predecessor->getTerminator(), block, next);
    collapseEscaping(next, block, dominance);
    propagateIfChanged(after, after->joinWith(next));
  }

  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const SyncLattice &before,
                                            SyncLattice *after) override {
    SyncState next = before.get();
    if (!regionFrom)
      transfer(branch.getOperation(), next);

    RegionSuccessor successor =
        regionTo ? RegionSuccessor(&branch->getRegion(*regionTo))
                 : RegionSuccessor(branch.getOperation());
    propagateRegionOperands(branch, regionFrom, successor, next);

    Block *target = branch->getBlock();
    if (regionTo) {
      Region &region = branch->getRegion(*regionTo);
      target = region.empty() ? branch->getBlock() : &region.front();
    }
    if (target)
      collapseEscaping(next, target, dominance);
    propagateIfChanged(after, after->joinWith(next));
  }

private:
  void transfer(Operation *operation, SyncState &state) {
    auto noEmit = [](Operation *, MutableArrayRef<IssueWait>) {};
    runTransfer(operation, state, noEmit);
  }

  void markCFGSuccessorsLive(Operation *operation, const SyncState &state) {
    if (operation->getNumSuccessors() == 0)
      return;
    Block *source = operation->getBlock();
    if (!source)
      return;
    for (Block *successor : operation->getSuccessors()) {
      SyncState next = state;
      propagateBranchOperands(operation, successor, next);
      collapseEscaping(next, successor, dominance);
      SyncLattice *blockState = getLattice(getProgramPointBefore(successor));
      propagateIfChanged(blockState, blockState->joinWith(next));
      Executable *blockLive =
          getOrCreate<Executable>(getProgramPointBefore(successor));
      propagateIfChanged(blockLive, blockLive->setToLive());
      Executable *edgeLive =
          getOrCreate<Executable>(getLatticeAnchor<CFGEdge>(source, successor));
      propagateIfChanged(edgeLive, edgeLive->setToLive());
    }
  }

  DominanceInfo &dominance;
};

static void emitWaits(OpBuilder &builder, Operation *operation,
                      MutableArrayRef<TokenWait> requirements) {
  builder.setInsertionPoint(operation);
  Type tokenType = MemTokenType::get(builder.getContext());
  llvm::sort(requirements, [](TokenWait lhs, TokenWait rhs) {
    if (lhs.sbid != rhs.sbid)
      return lhs.sbid < rhs.sbid;
    return lhs.mode < rhs.mode;
  });
  uint32_t destinationMask = 0;
  unsigned destinationCount = 0;
  for (TokenWait wait : requirements) {
    if (wait.mode != SWSBTokenMode::destination)
      continue;
    destinationMask |= uint32_t{1} << wait.sbid;
    ++destinationCount;
  }
  if (destinationCount > 1)
    SyncOp::create(builder, operation->getLoc(), tokenType, SyncKind::allwr,
                   Value(), destinationMask);
  for (TokenWait wait : requirements) {
    if (destinationCount > 1 && wait.mode == SWSBTokenMode::destination)
      continue;
    SyncOp sync = SyncOp::create(builder, operation->getLoc(), tokenType,
                                 SyncKind::nop, Value());
    FinalSWSB swsb;
    swsb.token = wait.sbid;
    swsb.tokenMode = wait.mode;
    sync.setFinalSWSB(swsb);
  }
}

static void collectBlocks(Region &region, SmallVectorImpl<Block *> &blocks) {
  for (Block &block : region) {
    blocks.push_back(&block);
    for (Operation &operation : block)
      for (Region &nested : operation.getRegions())
        collectBlocks(nested, blocks);
  }
}

static Operation *getDpasChainRoot(Operation *operation) {
  DpasOp dpas = dyn_cast<DpasOp>(operation);
  while (dpas) {
    DpasOp producer = dpas.getAcc().getDefiningOp<DpasOp>();
    if (!producer)
      break;
    dpas = producer;
  }
  return dpas ? dpas.getOperation() : operation;
}

static AllocationPlan buildAllocationPlan(func::FuncOp function,
                                          DataFlowSolver &solver,
                                          unsigned sbidCount) {
  SmallVector<Block *> blocks;
  collectBlocks(function.getBody(), blocks);
  llvm::MapVector<Operation *, SmallVector<Operation *>> groups;
  DenseMap<Operation *, SmallVector<Operation *, 8>> interference;

  for (Block *block : blocks) {
    for (Operation &operation : *block) {
      if (!isa<AsyncScoreboardOpInterface>(operation))
        continue;
      Operation *root = getDpasChainRoot(&operation);
      groups[root].push_back(&operation);
    }

    const SyncLattice *entry =
        solver.lookupState<SyncLattice>(solver.getProgramPointBefore(block));
    if (!entry || !entry->get().initialized)
      continue;
    for (Operation &operation : *block) {
      if (!isa<AsyncScoreboardOpInterface>(operation))
        continue;
      Operation *root = getDpasChainRoot(&operation);
      const SyncLattice *before = solver.lookupState<SyncLattice>(
          solver.getProgramPointBefore(&operation));
      if (!before || !before->get().initialized)
        continue;
      for (const IssueTicket &ticket : before->get().issues) {
        Operation *incomingRoot = getDpasChainRoot(ticket.issue);
        if (incomingRoot == root)
          continue;
        if (!llvm::is_contained(interference[root], incomingRoot))
          interference[root].push_back(incomingRoot);
        if (!llvm::is_contained(interference[incomingRoot], root))
          interference[incomingRoot].push_back(root);
      }
    }
  }

  DenseMap<Operation *, unsigned> groupTokens;
  for (auto &[root, members] : groups) {
    for (Operation *member : members) {
      FinalSWSB existing = cast<SWSBInfoOpInterface>(member).getFinalSWSB();
      if (existing.token < 0 ||
          static_cast<unsigned>(existing.token) >= sbidCount ||
          existing.tokenMode != SWSBTokenMode::set)
        continue;
      groupTokens[root] = existing.token;
      break;
    }
  }

  for (auto &[root, members] : groups) {
    if (groupTokens.contains(root))
      continue;
    uint32_t used = 0;
    for (Operation *neighbor : interference[root]) {
      auto token = groupTokens.find(neighbor);
      if (token != groupTokens.end())
        used |= uint32_t{1} << token->second;
    }
    unsigned selected = 0;
    while (selected < sbidCount && (used & (uint32_t{1} << selected)))
      ++selected;
    if (selected == sbidCount) {
      std::array<unsigned, 32> conflicts{};
      for (Operation *neighbor : interference[root]) {
        auto token = groupTokens.find(neighbor);
        if (token != groupTokens.end())
          ++conflicts[token->second];
      }
      selected = 0;
      for (unsigned candidate : llvm::seq<unsigned>(1, sbidCount))
        if (conflicts[candidate] < conflicts[selected])
          selected = candidate;
    }
    groupTokens[root] = selected;
  }

  AllocationPlan plan;
  for (auto &[root, members] : groups) {
    auto assigned = groupTokens.find(root);
    assert(assigned != groupTokens.end() &&
           "every reachable issue group must have an assigned SBID");
    unsigned token = assigned->second;
    for (Operation *operation : members) {
      plan[operation] = token;
      SWSBInfoOpInterface swsb = cast<SWSBInfoOpInterface>(operation);
      FinalSWSB final = swsb.getFinalSWSB();
      final.token = token;
      final.tokenMode = SWSBTokenMode::set;
      swsb.setFinalSWSB(final);
    }
  }
  return plan;
}

static LogicalResult validateExistingSynchronization(func::FuncOp function,
                                                     unsigned sbidCount) {
  uint32_t legalMask = sbidCount == 32 ? std::numeric_limits<uint32_t>::max()
                                       : (uint32_t{1} << sbidCount) - 1;
  DenseMap<Operation *, unsigned> assignedDpasChains;
  WalkResult result = function.walk([&](Operation *operation) {
    if (SyncOp sync = dyn_cast<SyncOp>(operation)) {
      FinalSWSB swsb = sync.getFinalSWSB();
      if (swsb.token >= 0 && static_cast<unsigned>(swsb.token) >= sbidCount) {
        sync.emitError("wait names SBID ")
            << swsb.token << " but this GRF mode exposes " << sbidCount;
        return WalkResult::interrupt();
      }
      if (sync.getSbidMask() & ~legalMask) {
        sync.emitError("selective wait mask names an unavailable SBID");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    DpasOp dpas = dyn_cast<DpasOp>(operation);
    if (!dpas)
      return WalkResult::advance();
    FinalSWSB current = dpas.getFinalSWSB();
    bool currentAssigned = current.token >= 0 &&
                           static_cast<unsigned>(current.token) < sbidCount &&
                           current.tokenMode == SWSBTokenMode::set;
    if (!currentAssigned)
      return WalkResult::advance();
    Operation *root = getDpasChainRoot(operation);
    auto [assigned, inserted] =
        assignedDpasChains.try_emplace(root, current.token);
    if (!inserted && assigned->second != static_cast<unsigned>(current.token)) {
      dpas.emitError("DPAS chain has inconsistent preassigned SBIDs");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

static void appendAllocationWaits(Operation *operation, const SyncState &state,
                                  const AllocationPlan &plan,
                                  SmallVectorImpl<IssueWait> &requirements) {
  if (!isa<AsyncScoreboardOpInterface>(operation))
    return;
  auto assigned = plan.find(operation);
  assert(assigned != plan.end() &&
         "reachable async issue must have an allocation");
  unsigned token = assigned->second;
  for (const IssueTicket &ticket : state.issues) {
    auto incoming = plan.find(ticket.issue);
    assert(incoming != plan.end() &&
           "live incoming issue must have an allocation");
    if (incoming->second != token)
      continue;
    if (DpasOp dpas = dyn_cast<DpasOp>(operation);
        dpas && isDpasChainPredecessor(ticket.issue, dpas))
      continue;
    requireWait(requirements, ticket,
                ticket.destinationPending ? SWSBTokenMode::destination
                                          : SWSBTokenMode::source);
  }
}

static void requireTokenWait(SmallVectorImpl<TokenWait> &requirements,
                             TokenWait wait) {
  if (wait.mode == SWSBTokenMode::source &&
      llvm::is_contained(requirements,
                         TokenWait{wait.sbid, SWSBTokenMode::destination}))
    return;
  if (wait.mode == SWSBTokenMode::destination)
    llvm::erase(requirements, TokenWait{wait.sbid, SWSBTokenMode::source});
  if (!llvm::is_contained(requirements, wait))
    requirements.push_back(wait);
}

static void
appendNextDestinationWaits(Operation *next, const SyncState &state,
                           SmallVectorImpl<IssueWait> &requirements) {
  if (!next || requirements.empty())
    return;
  for (IssueWait wait : computeRequirement(next, state))
    if (wait.mode == SWSBTokenMode::destination)
      requireWait(requirements, wait);
}

static void rewriteWithSolver(func::FuncOp function, DataFlowSolver &solver,
                              const AllocationPlan &plan) {
  OpBuilder builder(function.getContext());
  SmallVector<Block *> blocks;
  collectBlocks(function.getBody(), blocks);
  for (Block *block : blocks) {
    SyncState local;
    if (const SyncLattice *entry = solver.lookupState<SyncLattice>(
            solver.getProgramPointBefore(block)))
      local = entry->get();
    if (!local.initialized)
      continue;

    SmallVector<Operation *> operations;
    for (Operation &operation : *block)
      operations.push_back(&operation);
    for (auto [index, operation] : llvm::enumerate(operations)) {
      auto emit = [&](Operation *target, SmallVector<IssueWait> &requirements) {
        Operation *next =
            index + 1 < operations.size() ? operations[index + 1] : nullptr;
        appendNextDestinationWaits(next, local, requirements);
        appendAllocationWaits(target, local, plan, requirements);
        SmallVector<TokenWait> tokenWaits;
        for (IssueWait wait : requirements) {
          auto token = plan.find(wait.issue);
          assert(token != plan.end() &&
                 "reachable async issue must have an allocation");
          requireTokenWait(tokenWaits, TokenWait{token->second, wait.mode});
        }
        if (!tokenWaits.empty())
          emitWaits(builder, target, tokenWaits);
      };
      runTransfer(operation, local, emit);
      if (isa<RegionBranchOpInterface>(operation))
        if (const SyncLattice *post = solver.lookupState<SyncLattice>(
                solver.getProgramPointAfter(operation)))
          local = post->get();
    }
  }
}

struct DistanceAccess {
  Operation *producer;
  Xe2IssuePipe pipe;
  uint8_t age;
  SmallVector<RegisterSpan, 4> sources;
  SmallVector<RegisterSpan, 2> destinations;
};

struct DistanceState {
  SmallVector<DistanceAccess, 16> accesses;
  bool initialized = false;
  bool allPathsWroteAddressRegister = false;
};

class DistanceLattice : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistanceLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  const DistanceState &get() const { return state; }

  ChangeResult joinWith(const DistanceState &incoming) {
    if (!incoming.initialized)
      return ChangeResult::NoChange;
    if (!state.initialized) {
      state = incoming;
      return ChangeResult::Change;
    }

    bool changed = false;
    bool allPathsWrote = state.allPathsWroteAddressRegister &&
                         incoming.allPathsWroteAddressRegister;
    if (allPathsWrote != state.allPathsWroteAddressRegister) {
      state.allPathsWroteAddressRegister = allPathsWrote;
      changed = true;
    }
    for (const DistanceAccess &access : incoming.accesses) {
      auto existing =
          llvm::find_if(state.accesses, [&](const DistanceAccess &it) {
            return it.producer == access.producer;
          });
      if (existing == state.accesses.end()) {
        state.accesses.push_back(access);
        changed = true;
        continue;
      }
      assert(existing->pipe == access.pipe &&
             existing->sources == access.sources &&
             existing->destinations == access.destinations &&
             "one instruction must have one physical distance footprint");
      uint8_t age = std::min(existing->age, access.age);
      if (age != existing->age) {
        existing->age = age;
        changed = true;
      }
    }
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    return joinWith(static_cast<const DistanceLattice &>(rhs).state);
  }

  ChangeResult setEntryState() {
    DistanceState entry;
    entry.initialized = true;
    if (state.initialized && state.accesses.empty() &&
        !state.allPathsWroteAddressRegister)
      return ChangeResult::NoChange;
    state = entry;
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const override {
    os << "initialized=" << state.initialized
       << " accesses=" << state.accesses.size()
       << " wrote-a0=" << state.allPathsWroteAddressRegister;
  }

private:
  DistanceState state;
};

static std::optional<unsigned> getPipeIndex(Xe2IssuePipe pipe) {
  switch (pipe) {
  case Xe2IssuePipe::integer:
    return 0;
  case Xe2IssuePipe::floating:
    return 1;
  case Xe2IssuePipe::none:
  case Xe2IssuePipe::send:
  case Xe2IssuePipe::systolic:
    return std::nullopt;
  case Xe2IssuePipe::count:
    llvm_unreachable("issue pipe count is not a pipe");
  }
  llvm_unreachable("unknown Xe2 issue pipe");
}

static SWSBDistancePipe getSWSBDistancePipe(Xe2IssuePipe pipe) {
  if (pipe == Xe2IssuePipe::floating)
    return SWSBDistancePipe::floating;
  assert(pipe == Xe2IssuePipe::integer && "distance requires an ALU pipe");
  return SWSBDistancePipe::in_order;
}

static SmallVector<RegisterSpan, 4> getRegisterSpans(ValueRange values) {
  SmallVector<RegisterSpan, 4> spans;
  for (Value value : values)
    if (std::optional<RegisterSpan> span = getRegisterSpan(value))
      insertSpan(spans, *span);
  return spans;
}

static std::optional<int64_t> getElementBytes(Type type) {
  if (IntegerType integer = dyn_cast<IntegerType>(type))
    return llvm::divideCeil(integer.getWidth(), 8u);
  if (FloatType floating = dyn_cast<FloatType>(type))
    return llvm::divideCeil(floating.getWidth(), 8u);
  return std::nullopt;
}

static void appendElementSpan(SmallVectorImpl<RegisterSpan> &spans,
                              RegisterSpan storage, int64_t element,
                              int64_t elementBytes) {
  constexpr int64_t bytesPerGRF = 64;
  int64_t byteBegin = storage.begin + element * elementBytes;
  int64_t byteEnd = byteBegin + elementBytes;
  assert(byteBegin >= storage.begin && byteEnd <= storage.end &&
         "verified ALU region must fit its register storage");
  RegisterSpan span{byteBegin / bytesPerGRF * bytesPerGRF,
                    (byteEnd + bytesPerGRF - 1) / bytesPerGRF * bytesPerGRF};
  insertSpan(spans, span);
}

static SmallVector<RegisterSpan, 4> getALUSourceSpans(Operation *operation,
                                                      ALUOpInterface alu) {
  SmallVector<RegisterSpan, 4> spans;
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    std::optional<RegisterSpan> storage = getRegisterSpan(operand);
    if (!storage)
      continue;
    Type elementType = alu.getExplicitSourceElementType(index).value_or(
        alu.getInstructionElementType());
    std::optional<int64_t> elementBytes = getElementBytes(elementType);
    assert(elementBytes && "verified ALU element type must have a byte width");
    RegionAttr region = alu.getSourceRegion(index);
    int64_t vertical = region ? region.getVstride() : 1;
    int64_t width = region ? region.getWidth() : 1;
    int64_t horizontal = region ? region.getHstride() : 0;
    int64_t subregister = alu.getSourceSubregister(index);
    for (unsigned lane = 0; lane < alu.getExecutionSize(); ++lane) {
      int64_t element =
          subregister + lane / width * vertical + lane % width * horizontal;
      appendElementSpan(spans, *storage, element, *elementBytes);
    }
  }
  return spans;
}

static SmallVector<RegisterSpan, 4> getALUDestinationSpans(Operation *operation,
                                                           ALUOpInterface alu) {
  if (operation->getNumResults() == 0)
    return {};
  std::optional<RegisterSpan> storage =
      getRegisterSpan(operation->getResult(0));
  if (!storage)
    return {};
  if (cast<InstructionIssueOpInterface>(operation).getInstructionKind() ==
      MachineInstructionKind::cmp)
    return {*storage};

  std::optional<int64_t> elementBytes =
      getElementBytes(alu.getInstructionElementType());
  assert(elementBytes && "verified ALU element type must have a byte width");
  DstRegionAttr region = alu.getDestinationRegion();
  int64_t stride = region ? region.getHstride() : 1;
  int64_t subregister = alu.getDestinationSubregister();
  SmallVector<RegisterSpan, 4> spans;
  for (unsigned lane = 0; lane < alu.getExecutionSize(); ++lane)
    appendElementSpan(spans, *storage, subregister + lane * stride,
                      *elementBytes);
  return spans;
}

static bool anyOverlap(ArrayRef<RegisterSpan> lhs, ArrayRef<RegisterSpan> rhs) {
  return llvm::any_of(lhs, [&](RegisterSpan left) {
    return llvm::any_of(
        rhs, [&](RegisterSpan right) { return left.overlaps(right); });
  });
}

static LogicalResult transferDistance(Operation *operation,
                                      DistanceState &state, bool annotate,
                                      OpBuilder *builder = nullptr) {
  if (!state.initialized)
    return success();
  SWSBInfoOpInterface swsb = dyn_cast<SWSBInfoOpInterface>(operation);
  if (!swsb)
    return success();
  if (isa<SyncOp>(operation)) {
    return success();
  }

  FailureOr<Xe2InstructionTiming> timing = getXe2InstructionTiming(operation);
  if (failed(timing))
    return failure();
  Xe2IssuePipe distancePipe = timing->pipe;
  std::optional<unsigned> currentPipeIndex = getPipeIndex(distancePipe);
  ALUOpInterface alu = dyn_cast<ALUOpInterface>(operation);
  SmallVector<RegisterSpan, 4> sources =
      alu ? getALUSourceSpans(operation, alu)
          : getRegisterSpans(operation->getOperands());
  SmallVector<RegisterSpan, 4> destinations =
      alu ? getALUDestinationSpans(operation, alu)
          : getRegisterSpans(operation->getResults());

  int32_t youngestDistance = -1;
  SWSBDistancePipe pipe = SWSBDistancePipe::none;
  for (const DistanceAccess &access : state.accesses) {
    if (access.age > 7)
      continue;

    bool raw = anyOverlap(sources, access.destinations);
    bool crossPipe = !currentPipeIndex || distancePipe != access.pipe;
    bool waw = crossPipe && anyOverlap(destinations, access.destinations);
    bool war = crossPipe && anyOverlap(destinations, access.sources);
    if (!raw && !waw && !war)
      continue;

    SWSBDistancePipe producerPipe =
        waw || war || (currentPipeIndex && crossPipe)
            ? SWSBDistancePipe::all
            : getSWSBDistancePipe(access.pipe);
    if (youngestDistance < 0) {
      youngestDistance = access.age;
      pipe = producerPipe;
    } else {
      youngestDistance = std::min<int32_t>(youngestDistance, access.age);
      if (pipe != producerPipe)
        pipe = SWSBDistancePipe::all;
    }
  }

  bool writesAddressRegister =
      llvm::any_of(operation->getResultTypes(), [](Type type) {
        ARFType arf = dyn_cast<ARFType>(type);
        return arf && arf.getFile() == ARFFile::a0;
      });
  if (writesAddressRegister && !state.allPathsWroteAddressRegister) {
    pipe = SWSBDistancePipe::floating;
    youngestDistance = 1;
  }

  if (annotate) {
    if (isa<DpasOp>(operation) && youngestDistance >= 0) {
      assert(builder && "distance replay requires a builder");
      FinalSWSB distance;
      distance.pipe = pipe;
      distance.distance = youngestDistance;
      SyncOp sync;
      for (Operation *previous = operation->getPrevNode(); previous;
           previous = previous->getPrevNode()) {
        SyncOp candidate = dyn_cast<SyncOp>(previous);
        if (!candidate)
          break;
        if (candidate.getKind() == SyncKind::nop &&
            candidate.getFinalSWSB().token < 0 &&
            candidate.getFinalSWSB().pipe == distance.pipe &&
            candidate.getFinalSWSB().distance == distance.distance) {
          sync = candidate;
          break;
        }
      }
      if (!sync) {
        builder->setInsertionPoint(operation);
        sync = SyncOp::create(*builder, operation->getLoc(),
                              MemTokenType::get(builder->getContext()),
                              SyncKind::nop, Value());
        sync.setFinalSWSB(distance);
      }
      FinalSWSB final = swsb.getFinalSWSB();
      final.pipe = SWSBDistancePipe::none;
      final.distance = -1;
      swsb.setFinalSWSB(final);
    } else {
      FinalSWSB final = swsb.getFinalSWSB();
      if (final.tokenMode == SWSBTokenMode::set &&
          pipe == SWSBDistancePipe::floating)
        pipe = SWSBDistancePipe::all;
      final.pipe = pipe;
      final.distance = youngestDistance;
      swsb.setFinalSWSB(final);
    }
  }

  if (writesAddressRegister)
    state.allPathsWroteAddressRegister = true;
  if (!currentPipeIndex)
    return success();

  constexpr uint8_t expiredAge = 8;
  for (DistanceAccess &access : state.accesses)
    if (access.pipe == distancePipe)
      access.age = std::min<uint8_t>(access.age + 1, expiredAge);
  llvm::erase_if(state.accesses, [&](const DistanceAccess &access) {
    return access.producer == operation;
  });
  state.accesses.push_back(DistanceAccess{
      operation, distancePipe, 1, std::move(sources), std::move(destinations)});
  return success();
}

class DistanceAnalysis : public DenseForwardDataFlowAnalysis<DistanceLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistanceAnalysis)

  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    auto markRegions = [&](Operation *operation) {
      for (Region &region : operation->getRegions()) {
        for (Block &block : region) {
          Executable *blockLive =
              getOrCreate<Executable>(getProgramPointBefore(&block));
          propagateIfChanged(blockLive, blockLive->setToLive());
          Operation *terminator = block.getTerminator();
          if (!terminator)
            continue;
          for (Block *successor : terminator->getSuccessors()) {
            Executable *edgeLive = getOrCreate<Executable>(
                getLatticeAnchor<CFGEdge>(&block, successor));
            propagateIfChanged(edgeLive, edgeLive->setToLive());
          }
        }
      }
    };
    markRegions(top);
    top->walk(markRegions);
    return DenseForwardDataFlowAnalysis<DistanceLattice>::initialize(top);
  }

  void setToEntryState(DistanceLattice *lattice) override {
    propagateIfChanged(lattice, lattice->setEntryState());
  }

  LogicalResult visitOperation(Operation *operation,
                               const DistanceLattice &before,
                               DistanceLattice *after) override {
    DistanceState next = before.get();
    if (failed(transferDistance(operation, next, false)))
      return failure();
    propagateIfChanged(after, after->joinWith(next));
    return success();
  }

  void visitBlockTransfer(Block *, ProgramPoint *, Block *,
                          const DistanceLattice &before,
                          DistanceLattice *after) override {
    propagateIfChanged(after, after->joinWith(before.get()));
  }

  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface,
                                            std::optional<unsigned>,
                                            std::optional<unsigned>,
                                            const DistanceLattice &before,
                                            DistanceLattice *after) override {
    propagateIfChanged(after, after->joinWith(before.get()));
  }
};

static LogicalResult assignDistanceDependencies(func::FuncOp function) {
  DataFlowSolver solver;
  loadBaselineAnalyses(solver);
  solver.load<DistanceAnalysis>();
  if (failed(solver.initializeAndRun(function)))
    return failure();

  SmallVector<Block *> blocks;
  collectBlocks(function.getBody(), blocks);
  OpBuilder builder(function.getContext());
  for (Block *block : blocks) {
    DistanceState local;
    if (const DistanceLattice *entry = solver.lookupState<DistanceLattice>(
            solver.getProgramPointBefore(block)))
      local = entry->get();
    SmallVector<Operation *> operations;
    for (Operation &operation : *block)
      operations.push_back(&operation);
    for (Operation *operation : operations) {
      if (failed(transferDistance(operation, local, true, &builder)))
        return failure();
      if (isa<RegionBranchOpInterface>(operation))
        if (const DistanceLattice *post = solver.lookupState<DistanceLattice>(
                solver.getProgramPointAfter(operation)))
          local = post->get();
    }
  }
  return success();
}

class InsertSync : public inter::impl::InsertSyncBase<InsertSync> {
public:
  using InsertSyncBase::InsertSyncBase;

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;

    IntegerAttr grfCount =
        function->getAttrOfType<IntegerAttr>(kGrfCountAttrName);
    TargetAttr targetAttr =
        function->getAttrOfType<TargetAttr>(kTargetAttrName);
    llvm::Expected<TargetConfig> target =
        targetAttr ? TargetConfig::resolve(targetAttr)
        : !chip.empty()
            ? TargetConfig::resolve(chip)
            : llvm::Expected<TargetConfig>(llvm::createStringError(
                  "synchronization requires a target attribute or --chip "
                  "option"));
    if (!target) {
      function.emitError(llvm::toString(target.takeError()));
      return signalPassFailure();
    }
    unsigned sbidCount = target->getSbidCount(grfCount ? grfCount.getInt()
                                                       : target->getGrfCount());
    if (failed(validateExistingSynchronization(function, sbidCount)))
      return signalPassFailure();
    DominanceInfo dominance(function);
    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<SyncAnalysis>(dominance);
    if (failed(solver.initializeAndRun(function)))
      return signalPassFailure();
    AllocationPlan plan = buildAllocationPlan(function, solver, sbidCount);
    if (failed(assignDistanceDependencies(function)))
      return signalPassFailure();
    rewriteWithSolver(function, solver, plan);
  }
};

} // namespace
