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
#include "llvm/ADT/STLExtras.h"

#include <array>
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
  unsigned sbid;
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
    assert(existing.sbid == ticket.sbid && "one issue must have one SBID");
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

  ChangeResult reset() {
    if (state.issues.empty() && state.values.empty())
      return ChangeResult::NoChange;
    state = SyncState();
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const override {
    os << "issues=" << state.issues.size() << " values=" << state.values.size();
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

static bool isDpasChainPredecessor(Operation *candidate, DpasOp consumer);

// The may-lattice only grows. Inferred waits are therefore applied during
// local rewrite replay, not analysis; explicit sync operations remain part of
// analysis because filtering by a fixed wait kind is monotone.
enum class TransferMode { Analysis, Rewrite };

static std::optional<RegisterSpan> getRegisterSpan(Value value) {
  RegType type = dyn_cast<RegType>(value.getType());
  if (!type || type.getBaseGRF() < 0 || type.getWidthDwords() == 0)
    return std::nullopt;
  int64_t begin = type.getBaseGRF() * 16;
  return RegisterSpan{begin, begin + type.getWidthDwords()};
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

static void applyTokenWait(SyncState &state, TokenWait wait) {
  for (IssueTicket &ticket : state.issues) {
    if (ticket.sbid != wait.sbid)
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
    IssueTicket *issue = llvm::find_if(state.issues, [&](const IssueTicket &it) {
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

static void requireWait(SmallVectorImpl<TokenWait> &requirements,
                        const IssueTicket &ticket, SWSBTokenMode mode) {
  TokenWait wait{ticket.sbid, mode};
  if (mode == SWSBTokenMode::source &&
      llvm::is_contained(requirements,
                         TokenWait{ticket.sbid, SWSBTokenMode::destination}))
    return;
  if (mode == SWSBTokenMode::destination)
    llvm::erase(requirements,
                TokenWait{ticket.sbid, SWSBTokenMode::source});
  if (!llvm::is_contained(requirements, wait))
    requirements.push_back(wait);
}

static void requireValue(SmallVectorImpl<TokenWait> &requirements, Value value,
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
    if (ticket.destinationPending && llvm::any_of(ticket.destinations,
          [&](RegisterSpan destination) { return span->overlaps(destination); }))
      requireWait(requirements, ticket, SWSBTokenMode::destination);
}

static void requireDefinition(SmallVectorImpl<TokenWait> &requirements,
                              RegisterSpan definition,
                              const SyncState &state,
                              DpasOp chainedConsumer = nullptr) {
  for (const IssueTicket &ticket : state.issues) {
    if (chainedConsumer &&
        isDpasChainPredecessor(ticket.issue, chainedConsumer))
      continue;
    if (ticket.sourcePending && llvm::any_of(ticket.sources,
          [&](RegisterSpan source) { return definition.overlaps(source); }))
      requireWait(requirements, ticket, SWSBTokenMode::source);
    if (ticket.destinationPending && llvm::any_of(ticket.destinations,
          [&](RegisterSpan destination) {
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

static SmallVector<TokenWait> computeRequirement(Operation *operation,
                                                 const SyncState &state) {
  SmallVector<TokenWait> requirements;
  if (isa<ContinueIfOp>(operation)) {
    for (const IssueTicket &ticket : state.issues)
      requireWait(requirements, ticket,
                  ticket.destinationPending ? SWSBTokenMode::destination
                                            : SWSBTokenMode::source);
    return requirements;
  }
  if (isa<RegionBranchTerminatorOpInterface>(operation)) {
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
  if (auto async = dyn_cast<AsyncScoreboardOpInterface>(operation)) {
    FinalSWSB swsb = cast<SWSBInfoOpInterface>(operation).getFinalSWSB();
    assert(swsb.token >= 0 && "async issue must have an assigned SBID");
    for (const IssueTicket &ticket : state.issues) {
      if (ticket.sbid != static_cast<unsigned>(swsb.token))
        continue;
      if (DpasOp dpas = dyn_cast<DpasOp>(operation);
          dpas && isDpasChainPredecessor(ticket.issue, dpas))
        continue;
      requireWait(requirements, ticket,
                  ticket.destinationPending ? SWSBTokenMode::destination
                                            : SWSBTokenMode::source);
    }
  }
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
  FinalSWSB swsb = cast<SWSBInfoOpInterface>(operation.getOperation())
                       .getFinalSWSB();
  assert(swsb.token >= 0 && swsb.tokenMode == SWSBTokenMode::set &&
         "async issue must have an assigned SBID");
  IssueTicket issue{operation.getOperation(), static_cast<unsigned>(swsb.token)};
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
    insertValue(state, ValueTicket{result, issue.issue,
                                   token && !destination, destination});
  }
}

template <typename EmitFn>
static void applyDrain(Operation *operation, SyncState &state,
                        TransferMode mode, EmitFn emit) {
  SmallVector<TokenWait> requirements = computeRequirement(operation, state);
  emit(operation, requirements);
  for (TokenWait wait : requirements)
    applyTokenWait(state, wait);
}

static void observeSync(SyncOp sync, SyncState &state) {
  if (sync.getKind() == SyncKind::nop) {
    FinalSWSB swsb = sync.getFinalSWSB();
    if (swsb.token >= 0 && swsb.tokenMode != SWSBTokenMode::set)
      applyTokenWait(state,
                     TokenWait{static_cast<unsigned>(swsb.token),
                               swsb.tokenMode});
  } else if (sync.getKind() == SyncKind::allrd)
    applyReadWait(state);
  else if (sync.getKind() == SyncKind::allwr)
    applyWriteWait(state);
  else if (sync.getKind() == SyncKind::bar)
    applyReadWait(state);
  deriveResults(sync, state);
}

template <typename EmitFn>
static void runTransfer(Operation *operation, SyncState &state,
                        TransferMode mode, EmitFn emit) {
  if (SyncOp sync = dyn_cast<SyncOp>(operation)) {
    observeSync(sync, state);
    return;
  }

  if (isa<RegionBranchOpInterface>(operation)) {
    applyDrain(operation, state, mode, emit);
    return;
  }

  if (isa<RegionBranchTerminatorOpInterface, BranchOpInterface>(operation)) {
    applyDrain(operation, state, mode, emit);
    return;
  }

  if (!emitsMachineInstruction(operation)) {
    if (!operation->hasTrait<OpTrait::xemachine::CompletionFree>())
      deriveResults(operation, state);
    return;
  }

  applyDrain(operation, state, mode, emit);
  if (auto async = dyn_cast<AsyncScoreboardOpInterface>(operation))
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
    propagateIfChanged(lattice, lattice->reset());
  }

  LogicalResult visitOperation(Operation *operation, const SyncLattice &before,
                               SyncLattice *after) override {
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
    auto noEmit = [](Operation *, MutableArrayRef<TokenWait>) {};
    runTransfer(operation, state, TransferMode::Analysis, noEmit);
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
  for (TokenWait wait : requirements) {
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

static void rewriteWithSolver(func::FuncOp function, DataFlowSolver &solver) {
  OpBuilder builder(function.getContext());
  SmallVector<Block *> blocks;
  collectBlocks(function.getBody(), blocks);
  for (Block *block : blocks) {
    SyncState local;
    if (const SyncLattice *entry = solver.lookupState<SyncLattice>(
            solver.getProgramPointBefore(block)))
      local = entry->get();

    SmallVector<Operation *> operations;
    for (Operation &operation : *block)
      operations.push_back(&operation);
    for (Operation *operation : operations) {
      auto emit = [&](Operation *target, SmallVector<TokenWait> &requirements) {
        if (!requirements.empty())
          emitWaits(builder, target, requirements);
      };
      runTransfer(operation, local, TransferMode::Rewrite, emit);
      if (isa<RegionBranchOpInterface>(operation))
        if (const SyncLattice *post = solver.lookupState<SyncLattice>(
                solver.getProgramPointAfter(operation)))
          local = post->get();
    }
  }
}

static Operation *getMachineDefiningOperation(Value value) {
  Operation *definition = value.getDefiningOp();
  while (definition &&
         definition->hasTrait<OpTrait::xemachine::NoMachineInst>()) {
    if (definition->getNumOperands() != 1)
      return nullptr;
    definition = definition->getOperand(0).getDefiningOp();
  }
  return definition;
}

static void assignDistanceDependencies(func::FuncOp function) {
  DenseMap<Operation *, int32_t> instructionIndices;
  int32_t nextInstruction = 0;
  function.walk([&](Operation *operation) {
    if (operation->hasTrait<OpTrait::xemachine::NoMachineInst>() ||
        operation->hasTrait<OpTrait::xemachine::NoAsmEmission>())
      return;
    instructionIndices[operation] = nextInstruction++;
  });

  bool hasWrittenAddressRegister = false;
  function.walk([&](Operation *operation) {
    SWSBInfoOpInterface swsb = dyn_cast<SWSBInfoOpInterface>(operation);
    if (!swsb)
      return;

    FinalSWSB final = swsb.getFinalSWSB();
    int32_t youngestDistance = -1;
    SWSBDistancePipe pipe = SWSBDistancePipe::none;
    for (Value operand : operation->getOperands()) {
      if (isa<MemTokenType>(operand.getType()))
        continue;
      Operation *producer = getMachineDefiningOperation(operand);
      if (!producer || isa<AsyncScoreboardOpInterface>(producer))
        continue;
      auto producerIndex = instructionIndices.find(producer);
      auto consumerIndex = instructionIndices.find(operation);
      if (producerIndex == instructionIndices.end() ||
          consumerIndex == instructionIndices.end())
        continue;
      int32_t distance = consumerIndex->second - producerIndex->second;
      if (distance < 1 || distance > 7)
        continue;
      ALUOpInterface alu = dyn_cast<ALUOpInterface>(producer);
      SWSBDistancePipe producerPipe =
          alu && alu.getInstructionElementType().isF32()
              ? SWSBDistancePipe::floating
              : SWSBDistancePipe::in_order;
      if (youngestDistance < 0) {
        youngestDistance = distance;
        pipe = producerPipe;
      } else {
        youngestDistance = std::min(youngestDistance, distance);
        if (pipe != producerPipe)
          pipe = SWSBDistancePipe::all;
      }
    }

    if (final.tokenMode == SWSBTokenMode::set &&
        pipe == SWSBDistancePipe::floating)
      pipe = SWSBDistancePipe::all;
    final.pipe = pipe;
    final.distance = youngestDistance;

    if (!hasWrittenAddressRegister) {
      for (Type type : operation->getResultTypes()) {
        ARFType arf = dyn_cast<ARFType>(type);
        if (!arf || arf.getFile() != ARFFile::a0)
          continue;
        final.pipe = SWSBDistancePipe::floating;
        final.distance = 1;
        hasWrittenAddressRegister = true;
        break;
      }
    }
    swsb.setFinalSWSB(final);
  });
}

class InsertSync : public inter::impl::InsertSyncBase<InsertSync> {
public:
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;

    unsigned nextSBID = 0;
    function.walk([&](Operation *operation) {
      if (SyncOp sync = dyn_cast<SyncOp>(operation)) {
        if (sync.getKind() == SyncKind::allwr)
          nextSBID = 0;
        return;
      }
      AsyncScoreboardOpInterface issue =
          dyn_cast<AsyncScoreboardOpInterface>(operation);
      if (!issue)
        return;
      SWSBInfoOpInterface swsb = cast<SWSBInfoOpInterface>(operation);
      FinalSWSB final = swsb.getFinalSWSB();
      if (DpasOp dpas = dyn_cast<DpasOp>(operation)) {
        if (auto producer = dpas.getAcc().getDefiningOp<DpasOp>())
          final.token = producer.getFinalSWSB().token;
        else
          final.token = nextSBID++ % 32;
      } else {
        final.token = nextSBID++ % 32;
      }
      final.tokenMode = SWSBTokenMode::set;
      swsb.setFinalSWSB(final);
    });

    DominanceInfo dominance(function);
    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<SyncAnalysis>(dominance);
    if (failed(solver.initializeAndRun(function)))
      return signalPassFailure();
    rewriteWithSolver(function, solver);
    assignDistanceDependencies(function);
  }
};

} // namespace
