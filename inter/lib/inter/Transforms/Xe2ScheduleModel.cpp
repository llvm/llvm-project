// Xe2 hardware policy for the target-neutral machine scheduler.

#include "Xe2ScheduleModel.h"

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Dialect/XeMachine/IR/XeMachineAliasAnalysis.h"
#include "inter/Transforms/MachineScheduler.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>

using namespace mlir;
using namespace inter::xemachine;

namespace {

static Xe2DependencyKind getXe2DependencyKind(inter::MachineHazardKind kind) {
  switch (kind) {
  case inter::MachineHazardKind::raw:
    return Xe2DependencyKind::raw;
  case inter::MachineHazardKind::war:
    return Xe2DependencyKind::war;
  case inter::MachineHazardKind::waw:
    return Xe2DependencyKind::waw;
  case inter::MachineHazardKind::order:
    return Xe2DependencyKind::order;
  }
  llvm_unreachable("unknown machine dependency kind");
}

static uint64_t getArfResource(ARFType type) {
  return (static_cast<uint64_t>(type.getFile()) << 32) |
         static_cast<uint32_t>(type.getIndex());
}

static uint64_t getGrfResource(unsigned grf) {
  return (uint64_t{1} << 63) | grf;
}

static void
collectAliasDefinitions(Value root,
                        const DenseMap<Operation *, unsigned> &nodes,
                        const RegisterAliasAnalysis &aliases,
                        SmallVectorImpl<Operation *> &definitions) {
  SmallVector<Value, 4> pending{root};
  SmallPtrSet<Value, 16> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *definition = value.getDefiningOp();
    if (!definition)
      continue;

    bool followed = false;
    for (const RegisterAliasAnalysis::Alias &alias :
         aliases.getAliases(value)) {
      if (alias.owner != definition)
        continue;
      pending.push_back(alias.value);
      followed = true;
    }
    if (followed)
      continue;
    if (!definition->hasTrait<OpTrait::xemachine::NoMachineInst>() &&
        nodes.count(definition) && !llvm::is_contained(definitions, definition))
      definitions.push_back(definition);
  }
}

static bool isFullDestructiveContinuation(Operation *operation) {
  if (operation->hasTrait<OpTrait::xemachine::NoMachineInst>())
    return false;
  RegisterStorageAliasOpInterface aliasOp =
      dyn_cast<RegisterStorageAliasOpInterface>(operation);
  if (!aliasOp)
    return false;
  SmallVector<RegisterStorageAlias, 2> relations;
  aliasOp.getRegisterStorageAliases(relations);
  if (relations.size() != 1)
    return false;
  const RegisterStorageAlias &relation = relations.front();
  RegType storageType = dyn_cast<RegType>(relation.storage.getType());
  RegType aliasType = dyn_cast<RegType>(relation.alias.getType());
  return relation.destructive && relation.offset == 0 && storageType &&
         aliasType &&
         storageType.getWidthDwords() == aliasType.getWidthDwords() &&
         relation.storage.getDefiningOp() == operation;
}

static bool
requiresPinnedDefinition(Value root,
                         const DenseMap<Operation *, unsigned> &nodes,
                         const RegisterAliasAnalysis &aliases) {
  SmallVector<Value, 4> pending{root};
  SmallPtrSet<Value, 16> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (!nodes.count(user))
        return true;
      bool forwards = false;
      for (const RegisterAliasAnalysis::Alias &alias :
           aliases.getAliases(value)) {
        if (alias.owner != user)
          continue;
        if (alias.destructive) {
          if (isFullDestructiveContinuation(user)) {
            forwards = true;
            continue;
          }
          return true;
        }
        pending.push_back(alias.value);
        forwards = true;
      }
      if (!forwards)
        continue;
    }
  }
  return false;
}

struct PressureValue {
  SmallVector<unsigned, 4> definitions;
  SmallVector<unsigned, 4> uses;
  unsigned units = 0;
  bool liveIn = false;
  bool liveAfter = false;
};

struct PressureModel {
  SmallVector<PressureValue, 16> values;
};

static void collectPressureUses(Value value,
                                const DenseMap<Operation *, unsigned> &nodes,
                                const DenseMap<Operation *, int64_t> &positions,
                                int64_t regionBegin, int64_t regionEnd,
                                SmallVectorImpl<unsigned> &uses,
                                bool &liveBefore, bool &liveAfter,
                                SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (owner->hasTrait<OpTrait::xemachine::NoMachineInst>()) {
      bool forwarded = false;
      for (Value result : owner->getResults()) {
        if (!isa<RegType>(result.getType()))
          continue;
        collectPressureUses(result, nodes, positions, regionBegin, regionEnd,
                            uses, liveBefore, liveAfter, visited);
        forwarded = true;
      }
      if (forwarded)
        continue;
    }
    DenseMap<Operation *, unsigned>::const_iterator node = nodes.find(owner);
    if (node != nodes.end()) {
      if (!llvm::is_contained(uses, node->second))
        uses.push_back(node->second);
      continue;
    }
    int64_t position = positions.lookup(owner);
    liveBefore |= position < regionBegin;
    liveAfter |= position > regionEnd;
  }
}

static void collectTokenCompletionUses(
    Value token, const DenseMap<Operation *, unsigned> &nodes,
    const DenseMap<Operation *, int64_t> &positions, int64_t regionBegin,
    int64_t regionEnd, SmallVectorImpl<unsigned> &uses, bool &liveBefore,
    bool &liveAfter, SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(token).second) {
    liveAfter = true;
    return;
  }
  for (OpOperand &use : token.getUses()) {
    Operation *owner = use.getOwner();
    DenseMap<Operation *, unsigned>::const_iterator node = nodes.find(owner);
    if (node != nodes.end()) {
      if (!llvm::is_contained(uses, node->second))
        uses.push_back(node->second);
    } else {
      int64_t position = positions.lookup(owner);
      liveBefore |= position < regionBegin;
      liveAfter |= position > regionEnd;
    }
    if (!owner->hasTrait<OpTrait::xemachine::NoMachineInst>())
      continue;
    bool forwarded = false;
    for (Value result : owner->getResults()) {
      if (!isa<MemTokenType>(result.getType()))
        continue;
      collectTokenCompletionUses(result, nodes, positions, regionBegin,
                                 regionEnd, uses, liveBefore, liveAfter,
                                 visited);
      forwarded = true;
    }
    if (!forwarded && isa<YieldOp, ContinueIfOp>(owner))
      liveAfter = true;
  }
  visited.erase(token);
}

static bool extendsPayloadToTokenCompletion(Operation *operation) {
  return isa<AsyncScoreboardOpInterface>(operation);
}

static PressureModel
buildPressureModel(ArrayRef<Operation *> operations,
                   const RegisterAliasAnalysis &aliases,
                   const DenseMap<Operation *, int64_t> &positions) {
  DenseMap<Operation *, unsigned> nodes;
  for (auto [index, operation] : llvm::enumerate(operations))
    nodes.try_emplace(operation, index);
  int64_t regionBegin = positions.lookup(operations.front());
  int64_t regionEnd = positions.lookup(operations.back());

  llvm::SmallSetVector<unsigned, 16> components;
  for (Operation *operation : operations) {
    for (Value result : operation->getResults()) {
      RegType reg = dyn_cast<RegType>(result.getType());
      if (!reg || reg.getWidthDwords() == 0)
        continue;
      const RegisterAliasAnalysis::ValueInfo *valueInfo =
          aliases.lookup(result);
      assert(valueInfo && "register result is missing alias information");
      components.insert(valueInfo->component);
    }
    for (Value operand : operation->getOperands()) {
      RegType reg = dyn_cast<RegType>(operand.getType());
      if (!reg || reg.getWidthDwords() == 0)
        continue;
      const RegisterAliasAnalysis::ValueInfo *valueInfo =
          aliases.lookup(operand);
      assert(valueInfo && "register operand is missing alias information");
      components.insert(valueInfo->component);
    }
  }

  PressureModel model;
  for (unsigned componentIndex : components) {
    const RegisterAliasAnalysis::Component &component =
        aliases.getComponents()[componentIndex];
    PressureValue &pressure = model.values.emplace_back();
    pressure.units = llvm::divideCeil(component.widthDwords, int64_t{16});
    for (Value value : component.members) {
      Operation *definition = value.getDefiningOp();
      if (definition &&
          !definition->hasTrait<OpTrait::xemachine::NoMachineInst>()) {
        DenseMap<Operation *, unsigned>::const_iterator node =
            nodes.find(definition);
        if (node != nodes.end()) {
          if (!llvm::is_contained(pressure.definitions, node->second))
            pressure.definitions.push_back(node->second);
        } else {
          int64_t position = positions.lookup(definition);
          pressure.liveIn |= position < regionBegin;
          pressure.liveAfter |= position > regionEnd;
        }
      } else if (!definition) {
        pressure.liveIn = true;
      }

      SmallPtrSet<Value, 16> visited;
      SmallVector<unsigned, 4> uses;
      bool liveBefore = false;
      bool liveAfter = false;
      collectPressureUses(value, nodes, positions, regionBegin, regionEnd, uses,
                          liveBefore, liveAfter, visited);
      for (OpOperand &use : value.getUses()) {
        Operation *owner = use.getOwner();
        if (!extendsPayloadToTokenCompletion(owner))
          continue;
        SmallPtrSet<Value, 8> visitedTokens;
        for (Value result : owner->getResults())
          if (isa<MemTokenType>(result.getType()))
            collectTokenCompletionUses(result, nodes, positions, regionBegin,
                                       regionEnd, uses, liveBefore, liveAfter,
                                       visitedTokens);
      }
      for (unsigned use : uses) {
        if (!llvm::is_contained(pressure.uses, use))
          pressure.uses.push_back(use);
        if (isa_and_nonnull<ArchRegOp>(definition) &&
            !llvm::is_contained(pressure.definitions, use))
          pressure.definitions.push_back(use);
      }
      pressure.liveIn |= liveBefore;
      pressure.liveAfter |= liveAfter;
    }
  }
  return model;
}

static unsigned getPeakPressure(ArrayRef<unsigned> order,
                                ArrayRef<PressureValue> values,
                                unsigned nodeCount) {
  BitVector scheduled(nodeCount);
  BitVector live(values.size());
  unsigned pressure = 0;
  auto makeLive = [&](unsigned index) {
    if (live.test(index))
      return;
    live.set(index);
    pressure += values[index].units;
  };
  auto makeDead = [&](unsigned index) {
    assert(live.test(index) && "pressure value is not live");
    live.reset(index);
    assert(pressure >= values[index].units && "pressure count underflow");
    pressure -= values[index].units;
  };
  for (auto [index, value] : llvm::enumerate(values)) {
    if (!value.liveIn)
      continue;
    makeLive(index);
  }
  unsigned peak = pressure;
  for (unsigned node : order) {
    for (auto [index, value] : llvm::enumerate(values)) {
      if (!llvm::is_contained(value.definitions, node))
        continue;
      makeLive(index);
    }
    peak = std::max(peak, pressure);
    scheduled.set(node);
    for (auto [index, value] : llvm::enumerate(values)) {
      if (!live.test(index) || value.liveAfter)
        continue;
      if (llvm::any_of(value.definitions, [&](unsigned definition) {
            return !scheduled.test(definition);
          }))
        continue;
      if (llvm::any_of(value.uses,
                       [&](unsigned use) { return !scheduled.test(use); }))
        continue;
      makeDead(index);
    }
  }
  return peak;
}

class Xe2RegionSession final : public inter::MachineScheduleRegionSession {
public:
  Xe2RegionSession(ArrayRef<Operation *> operations,
                   const RegisterAliasAnalysis &aliases,
                   const DenseMap<Operation *, int64_t> &positions)
      : pressureModel(buildPressureModel(operations, aliases, positions)),
        nodeCount(operations.size()) {
    SmallVector<unsigned, 16> original;
    llvm::append_range(original, llvm::seq<unsigned>(nodeCount));
    originalPeak = getPeakPressure(original, pressureModel.values, nodeCount);
  }

  bool canSchedulePrefix(ArrayRef<unsigned> prefix) const override {
    unsigned peak = getPeakPressure(prefix, pressureModel.values, nodeCount);
    return peak <= originalPeak;
  }

private:
  PressureModel pressureModel;
  unsigned nodeCount;
  unsigned originalPeak;
};

class Xe2ScheduleModel final : public inter::MachineScheduleModel {
public:
  Xe2ScheduleModel(RegisterAliasAnalysis aliases,
                   DenseMap<Operation *, int64_t> positions)
      : aliases(std::move(aliases)), positions(std::move(positions)) {}

  bool isSchedulable(Operation *operation) const override {
    if (!isa<InstructionIssueOpInterface>(operation) ||
        operation->hasTrait<OpTrait::xemachine::FullScoreboardDrain>())
      return false;
    SendOp send = dyn_cast<SendOp>(operation);
    return !send || !send.getEot();
  }

  bool isNoInstruction(Operation *operation) const override {
    return operation->hasTrait<OpTrait::xemachine::NoMachineInst>();
  }

  bool isSupportedRegionOperation(Operation *operation) const override {
    return isa<ExecIfOp, UniformIfOp, UniformLoopOp, PayloadPrologueOp>(
        operation);
  }

  inter::MachineHazardKind
  classifyDataDependency(Value operand) const override {
    return isa<MemTokenType>(operand.getType())
               ? inter::MachineHazardKind::order
               : inter::MachineHazardKind::raw;
  }

  void getStorageAccesses(
      Operation *operation,
      SmallVectorImpl<inter::MachineStorageAccess> &accesses) const override {
    if (operation->hasTrait<OpTrait::xemachine::NoMachineInst>())
      return;
    auto record = [&](uint64_t resource, bool read, bool write) {
      for (inter::MachineStorageAccess &access : accesses) {
        if (access.resource != resource)
          continue;
        access.read |= read;
        access.write |= write;
        return;
      }
      accesses.push_back({resource, read, write});
    };
    auto recordValue = [&](Value value, bool read, bool write) {
      if (ARFType arf = dyn_cast<ARFType>(value.getType())) {
        record(getArfResource(arf), read, write);
        return;
      }
      RegType reg = dyn_cast<RegType>(value.getType());
      if (!reg || reg.getWidthDwords() == 0)
        return;
      int64_t firstDword = static_cast<int64_t>(reg.getBaseGRF()) * 16;
      const RegisterAliasAnalysis::ValueInfo *valueInfo = aliases.lookup(value);
      if (valueInfo) {
        const RegisterAliasAnalysis::Component &component =
            aliases.getComponents()[valueInfo->component];
        if (!component.fixedOriginDwords)
          return;
        firstDword = *component.fixedOriginDwords + valueInfo->offsetDwords;
      } else if (reg.getBaseGRF() < 0) {
        return;
      }
      int64_t firstGRF = firstDword / 16;
      int64_t grfEnd = llvm::divideCeil(
          firstDword + static_cast<int64_t>(reg.getWidthDwords()), int64_t{16});
      for (int64_t grf : llvm::seq(firstGRF, grfEnd))
        record(getGrfResource(grf), read, write);
    };
    for (Value operand : operation->getOperands())
      recordValue(operand, /*read=*/true, /*write=*/false);
    for (Value result : operation->getResults())
      recordValue(result, /*read=*/false, /*write=*/true);
  }

  void getExtraDependencies(ArrayRef<Operation *> operations,
                            SmallVectorImpl<inter::MachineExtraDependency>
                                &dependencies) const override {
    DenseMap<Operation *, unsigned> nodes;
    for (auto [index, operation] : llvm::enumerate(operations))
      nodes.try_emplace(operation, index);

    for (auto [index, operation] : llvm::enumerate(operations)) {
      if (operation->hasTrait<OpTrait::xemachine::NoMachineInst>() ||
          !llvm::any_of(operation->getResults(), [&](Value result) {
            return requiresPinnedDefinition(result, nodes, aliases);
          }))
        continue;
      for (unsigned predecessor : llvm::seq<unsigned>(index))
        dependencies.push_back({operations[predecessor], operation,
                                inter::MachineHazardKind::order});
    }

    UniformLoopOp loop =
        dyn_cast<UniformLoopOp>(operations.front()->getBlock()->getParentOp());
    if (!loop)
      return;
    ContinueIfOp terminator =
        dyn_cast<ContinueIfOp>(operations.front()->getBlock()->getTerminator());
    if (!terminator)
      return;
    for (auto [argument, carried] :
         llvm::zip_equal(operations.front()->getBlock()->getArguments(),
                         terminator.getCarried())) {
      if (!isa<RegType, ARFType>(argument.getType()))
        continue;
      SmallVector<Operation *, 4> definitions;
      collectAliasDefinitions(carried, nodes, aliases, definitions);
      if (definitions.empty())
        continue;

      SmallVector<Value, 4> pending{argument};
      SmallPtrSet<Value, 16> visited;
      while (!pending.empty()) {
        Value value = pending.pop_back_val();
        if (!visited.insert(value).second)
          continue;
        for (OpOperand &use : value.getUses()) {
          Operation *user = use.getOwner();
          if (isa<RegisterStorageAliasOpInterface>(user)) {
            llvm::append_range(pending, user->getResults());
            continue;
          }
          if (!nodes.count(user))
            continue;
          for (Operation *definition : definitions)
            if (user != definition)
              dependencies.push_back(
                  {user, definition, inter::MachineHazardKind::war});
        }
      }
    }
  }

  std::unique_ptr<inter::MachineScheduleRegionSession>
  createRegionSession(ArrayRef<Operation *> operations) const override {
    return std::make_unique<Xe2RegionSession>(operations, aliases, positions);
  }

  std::unique_ptr<inter::MachineScheduleState> createState() const override;

private:
  RegisterAliasAnalysis aliases;
  DenseMap<Operation *, int64_t> positions;
};

class Xe2ScheduleState final : public inter::MachineScheduleState {
public:
  FailureOr<inter::MachineScheduleIssue> previewIssue(
      Operation *operation,
      ArrayRef<inter::MachineScheduleDependency> dependencies) const override {
    FailureOr<Xe2InstructionTiming> timing = getXe2InstructionTiming(operation);
    if (failed(timing))
      return failure();

    int64_t dependencyReadyCycle = 0;
    for (const inter::MachineScheduleDependency &dependency : dependencies) {
      int64_t readyCycle = dependency.issue.cycle;
      if (dependency.issue.instruction) {
        FailureOr<Xe2InstructionTiming> producerTiming =
            getXe2InstructionTiming(dependency.producer);
        if (failed(producerTiming))
          return failure();
        readyCycle += getXe2RequiredGap(*producerTiming,
                                        getXe2DependencyKind(dependency.kind));
      }
      dependencyReadyCycle = std::max(dependencyReadyCycle, readyCycle);
    }

    inter::MachineScheduleIssue issue;
    issue.instruction = timing->issueClass != InstructionIssueClass::none;
    if (!issue.instruction) {
      issue.cycle = dependencyReadyCycle;
      issue.nextCycle = currentCycle;
      return issue;
    }

    unsigned pipe = static_cast<unsigned>(timing->pipe);
    issue.cycle =
        std::max({currentCycle, dependencyReadyCycle, pipeReadyCycle[pipe]});
    issue.nextCycle = issue.cycle + 1;
    issue.stallCycles = issue.cycle - currentCycle;
    return issue;
  }

  FailureOr<inter::MachineScheduleIssue> commitIssue(
      Operation *operation,
      ArrayRef<inter::MachineScheduleDependency> dependencies) override {
    FailureOr<inter::MachineScheduleIssue> issue =
        previewIssue(operation, dependencies);
    if (failed(issue) || !issue->instruction)
      return issue;
    FailureOr<Xe2InstructionTiming> timing = getXe2InstructionTiming(operation);
    if (failed(timing))
      return failure();
    currentCycle = issue->nextCycle;
    unsigned pipe = static_cast<unsigned>(timing->pipe);
    pipeReadyCycle[pipe] = issue->cycle + timing->occupancy;
    return issue;
  }

  FailureOr<bool>
  canFill(Operation *baseline,
          ArrayRef<inter::MachineScheduleDependency> baselineDependencies,
          Operation *candidate,
          ArrayRef<inter::MachineScheduleDependency> candidateDependencies)
      const override {
    FailureOr<inter::MachineScheduleIssue> baselineBefore =
        previewIssue(baseline, baselineDependencies);
    FailureOr<inter::MachineScheduleIssue> candidatePreview =
        previewIssue(candidate, candidateDependencies);
    if (failed(baselineBefore) || failed(candidatePreview))
      return failure();
    if (!candidatePreview->instruction ||
        candidatePreview->cycle >= baselineBefore->cycle)
      return false;
    if (candidatePreview->stallCycles != 0 &&
        (!isFullDestructiveContinuation(baseline) ||
         !isFullDestructiveContinuation(candidate)))
      return false;
    Xe2ScheduleState trial = *this;
    if (failed(trial.commitIssue(candidate, candidateDependencies)))
      return failure();
    FailureOr<inter::MachineScheduleIssue> baselineAfter =
        trial.previewIssue(baseline, baselineDependencies);
    if (failed(baselineAfter))
      return failure();
    return baselineAfter->cycle <= baselineBefore->cycle;
  }

private:
  int64_t currentCycle = 0;
  std::array<int64_t, static_cast<unsigned>(Xe2IssuePipe::count)>
      pipeReadyCycle{};
};

std::unique_ptr<inter::MachineScheduleState>
Xe2ScheduleModel::createState() const {
  return std::make_unique<Xe2ScheduleState>();
}

} // namespace

FailureOr<std::unique_ptr<inter::MachineScheduleModel>>
inter::createXe2ScheduleModel(func::FuncOp function) {
  FailureOr<RegisterAliasAnalysis> aliases =
      RegisterAliasAnalysis::create(function);
  if (failed(aliases))
    return failure();
  DenseMap<Operation *, int64_t> positions;
  int64_t nextPosition = 0;
  function.walk(
      [&](Operation *operation) { positions[operation] = nextPosition++; });
  std::unique_ptr<inter::MachineScheduleModel> model =
      std::make_unique<Xe2ScheduleModel>(std::move(*aliases),
                                         std::move(positions));
  return model;
}
