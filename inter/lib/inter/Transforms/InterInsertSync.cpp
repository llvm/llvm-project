// Materialize conservative Xe scoreboard waits from a dense machine-state
// analysis. Token pseudos remain in the IR as zero-byte bookkeeping.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

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

struct CompletionTicket {
  Value id;
  std::optional<RegisterSpan> span;
  bool read = false;
  bool write = false;
};

struct SourceTicket {
  RegisterSpan span;
  bool completedByAllWr;
};

struct SyncState {
  SmallVector<CompletionTicket, 8> completions;
  SmallVector<SourceTicket, 8> sources;
};

static bool sameCompletionKey(const CompletionTicket &lhs,
                              const CompletionTicket &rhs) {
  if (lhs.id || rhs.id)
    return lhs.id == rhs.id;
  return lhs.span == rhs.span;
}

static bool insertCompletion(SyncState &state, CompletionTicket ticket) {
  for (CompletionTicket &existing : state.completions) {
    if (!sameCompletionKey(existing, ticket))
      continue;
    bool changed = false;
    if (!existing.span && ticket.span) {
      existing.span = ticket.span;
      changed = true;
    }
    if (ticket.read && !existing.read) {
      existing.read = true;
      changed = true;
    }
    if (ticket.write && !existing.write) {
      existing.write = true;
      changed = true;
    }
    return changed;
  }
  state.completions.push_back(ticket);
  return true;
}

static bool insertSource(SyncState &state, SourceTicket ticket) {
  for (SourceTicket &existing : state.sources) {
    if (existing.span != ticket.span)
      continue;
    bool completedByAllWr =
        existing.completedByAllWr && ticket.completedByAllWr;
    if (completedByAllWr == existing.completedByAllWr)
      return false;
    existing.completedByAllWr = completedByAllWr;
    return true;
  }
  state.sources.push_back(ticket);
  return true;
}

class SyncLattice : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SyncLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  const SyncState &get() const { return state; }

  ChangeResult joinWith(const SyncState &incoming) {
    bool changed = false;
    for (const CompletionTicket &ticket : incoming.completions)
      changed |= insertCompletion(state, ticket);
    for (const SourceTicket &ticket : incoming.sources)
      changed |= insertSource(state, ticket);
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    return joinWith(static_cast<const SyncLattice &>(rhs).state);
  }

  ChangeResult reset() {
    if (state.completions.empty() && state.sources.empty())
      return ChangeResult::NoChange;
    state = SyncState();
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const override {
    os << "completions=" << state.completions.size()
       << " sources=" << state.sources.size();
  }

private:
  SyncState state;
};

struct WaitRequirement {
  bool read = false;
  bool write = false;

  bool hasWait() const { return read || write; }
};

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

// Values that do not dominate an edge target become anonymous tickets. Their
// physical footprint and completion class still participate in later hazards.
static void collapseEscaping(SyncState &state, Block *target,
                             DominanceInfo &dominance) {
  SmallVector<CompletionTicket, 8> kept;
  auto mergeIntoKept = [&](CompletionTicket ticket) {
    for (CompletionTicket &existing : kept) {
      if (!sameCompletionKey(existing, ticket))
        continue;
      if (!existing.span && ticket.span)
        existing.span = ticket.span;
      existing.read |= ticket.read;
      existing.write |= ticket.write;
      return;
    }
    kept.push_back(ticket);
  };
  for (const CompletionTicket &ticket : state.completions) {
    if (!ticket.id) {
      mergeIntoKept(ticket);
      continue;
    }
    Block *definition = getDefiningBlock(ticket.id);
    if (!definition || dominance.dominates(definition, target)) {
      mergeIntoKept(ticket);
      continue;
    }
    CompletionTicket escaping = ticket;
    escaping.id = Value();
    mergeIntoKept(escaping);
  }
  state.completions = std::move(kept);
}

static void applyReadWait(SyncState &state) {
  for (CompletionTicket &ticket : state.completions)
    ticket.read = false;
  llvm::erase_if(state.completions, [](const CompletionTicket &ticket) {
    return !ticket.read && !ticket.write;
  });
  state.sources.clear();
}

static void applyWriteWait(SyncState &state) {
  for (CompletionTicket &ticket : state.completions)
    ticket.write = false;
  llvm::erase_if(state.completions, [](const CompletionTicket &ticket) {
    return !ticket.read && !ticket.write;
  });
  llvm::erase_if(state.sources, [](const SourceTicket &ticket) {
    return ticket.completedByAllWr;
  });
}

static void applyWait(SyncState &state, const WaitRequirement &requirement) {
  if (requirement.read)
    applyReadWait(state);
  if (requirement.write)
    applyWriteWait(state);
}

static void requireValue(WaitRequirement &requirement, Value value,
                         const SyncState &state) {
  std::optional<RegisterSpan> span = getRegisterSpan(value);
  for (const CompletionTicket &ticket : state.completions) {
    if (ticket.id == value) {
      requirement.read |= ticket.read;
      requirement.write |= ticket.write;
    }
    if (span && ticket.write && ticket.span && span->overlaps(*ticket.span))
      requirement.write = true;
  }
}

static void requireDefinition(WaitRequirement &requirement,
                              RegisterSpan definition, const SyncState &state) {
  for (const SourceTicket &ticket : state.sources) {
    if (!definition.overlaps(ticket.span))
      continue;
    if (ticket.completedByAllWr)
      requirement.write = true;
    else
      requirement.read = true;
  }
  for (const CompletionTicket &ticket : state.completions)
    if (ticket.write && ticket.span && definition.overlaps(*ticket.span))
      requirement.write = true;
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

static bool hasMemTokenResult(Operation *operation) {
  return llvm::any_of(operation->getResultTypes(),
                      [](Type type) { return isa<MemTokenType>(type); });
}

static bool isAsyncMessage(Operation *operation) {
  if (!emitsMachineInstruction(operation) || isFullDrain(operation) ||
      !hasMemTokenResult(operation))
    return false;
  MemoryEffectOpInterface effects =
      dyn_cast<MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return !instances.empty();
}

static WaitRequirement computeRequirement(Operation *operation,
                                          const SyncState &state) {
  WaitRequirement requirement;
  if (isFullDrain(operation)) {
    for (const CompletionTicket &ticket : state.completions) {
      requirement.read |= ticket.read;
      requirement.write |= ticket.write;
    }
    for (const SourceTicket &ticket : state.sources) {
      if (ticket.completedByAllWr)
        requirement.write = true;
      else
        requirement.read = true;
    }
    return requirement;
  }

  if (!emitsMachineInstruction(operation))
    return requirement;

  for (OpOperand &operand : operation->getOpOperands()) {
    if (isControlFlowOp(operation) && isForwardedControlOperand(operand))
      continue;
    requireValue(requirement, operand.get(), state);
  }

  if (isa<RegionBranchOpInterface>(operation))
    return requirement;
  for (Value result : operation->getResults())
    if (std::optional<RegisterSpan> span = getRegisterSpan(result))
      requireDefinition(requirement, *span, state);
  return requirement;
}

static void deriveValue(SyncState &state, ValueRange sources,
                        Value destination) {
  CompletionTicket derived;
  derived.id = destination;
  derived.span = getRegisterSpan(destination);
  for (Value source : sources) {
    for (const CompletionTicket &ticket : state.completions) {
      if (ticket.id != source)
        continue;
      derived.read |= ticket.read;
      derived.write |= ticket.write;
    }
  }
  if (derived.read || derived.write)
    insertCompletion(state, derived);
}

static void deriveValue(SyncState &state, Value source, Value destination) {
  deriveValue(state, ValueRange(source), destination);
}

static void deriveResults(Operation *operation, SyncState &state) {
  for (Value result : operation->getResults())
    deriveValue(state, operation->getOperands(), result);
}

static void recordIssue(Operation *operation, SyncState &state) {
  bool writesDestination =
      llvm::any_of(operation->getResults(), [](Value value) {
        RegType type = dyn_cast<RegType>(value.getType());
        return type && type.getWidthDwords() != 0;
      });

  for (Value result : operation->getResults()) {
    CompletionTicket ticket;
    ticket.id = result;
    ticket.span = getRegisterSpan(result);
    ticket.read = !writesDestination;
    ticket.write = writesDestination;
    insertCompletion(state, ticket);
  }

  for (Value operand : operation->getOperands())
    if (std::optional<RegisterSpan> span = getRegisterSpan(operand))
      insertSource(state, SourceTicket{*span, writesDestination});
}

template <typename EmitFn>
static void applyDrain(Operation *operation, SyncState &state,
                       TransferMode mode, EmitFn emit) {
  WaitRequirement requirement = computeRequirement(operation, state);
  emit(operation, requirement);
  if (mode == TransferMode::Rewrite)
    applyWait(state, requirement);
}

static void observeSync(SyncOp sync, SyncState &state) {
  if (sync.getKind() == SyncKind::allrd)
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
  if (isAsyncMessage(operation))
    recordIssue(operation, state);
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
    auto noEmit = [](Operation *, const WaitRequirement &) {};
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
                      const WaitRequirement &requirement) {
  builder.setInsertionPoint(operation);
  Type tokenType = MemTokenType::get(builder.getContext());
  Value dependency;
  if (requirement.read) {
    SyncOp sync = SyncOp::create(
        builder, operation->getLoc(), tokenType,
        SyncKindAttr::get(builder.getContext(), SyncKind::allrd), dependency);
    dependency = sync.getToken();
  }
  if (requirement.write)
    SyncOp::create(builder, operation->getLoc(), tokenType,
                   SyncKindAttr::get(builder.getContext(), SyncKind::allwr),
                   dependency);
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
      auto emit = [&](Operation *target, const WaitRequirement &requirement) {
        if (requirement.hasWait())
          emitWaits(builder, target, requirement);
      };
      runTransfer(operation, local, TransferMode::Rewrite, emit);
      if (isa<RegionBranchOpInterface>(operation))
        if (const SyncLattice *post = solver.lookupState<SyncLattice>(
                solver.getProgramPointAfter(operation)))
          local = post->get();
    }
  }
}

class InsertSync : public inter::impl::InsertSyncBase<InsertSync> {
public:
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;

    DominanceInfo dominance(function);
    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<SyncAnalysis>(dominance);
    if (failed(solver.initializeAndRun(function)))
      return signalPassFailure();
    rewriteWithSolver(function, solver);
  }
};

} // namespace
