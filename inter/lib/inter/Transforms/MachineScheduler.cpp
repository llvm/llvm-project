// Target-neutral deterministic original-order gap-filling scheduler.

#include "inter/Transforms/MachineScheduler.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>

using namespace mlir;

namespace {

using inter::MachineExtraDependency;
using inter::MachineHazardKind;
using inter::MachineScheduleDependency;
using inter::MachineScheduleIssue;
using inter::MachineScheduleModel;
using inter::MachineScheduleRegionSession;
using inter::MachineScheduleState;
using inter::MachineStorageAccess;

struct CollectedRegion {
  SmallVector<Operation *, 16> operations;
  Block *block = nullptr;
};

struct ScheduleEdge {
  unsigned source;
  unsigned target;
  MachineHazardKind kind;
};

struct ScheduleGraph {
  DenseMap<Operation *, unsigned> nodes;
  SmallVector<ScheduleEdge, 32> edges;
  SmallVector<SmallVector<unsigned, 4>, 16> incoming;
  SmallVector<SmallVector<unsigned, 4>, 16> outgoing;
  SmallVector<unsigned, 16> pendingPredecessors;
  bool legal = true;
};

struct StorageState {
  std::optional<unsigned> lastWriter;
  SmallVector<unsigned, 4> readers;
};

static void addEdge(ScheduleGraph &graph, unsigned source, unsigned target,
                    MachineHazardKind kind) {
  if (source == target)
    return;
  for (const ScheduleEdge &edge : graph.edges)
    if (edge.source == source && edge.target == target && edge.kind == kind)
      return;
  unsigned index = graph.edges.size();
  graph.edges.push_back({source, target, kind});
  graph.incoming[target].push_back(index);
  graph.outgoing[source].push_back(index);
  ++graph.pendingPredecessors[target];
}

static void addStorageEdges(ArrayRef<Operation *> operations,
                            const MachineScheduleModel &model,
                            ScheduleGraph &graph) {
  DenseMap<uint64_t, StorageState> states;
  for (auto [index, operation] : llvm::enumerate(operations)) {
    SmallVector<MachineStorageAccess, 4> accesses;
    model.getStorageAccesses(operation, accesses);
    llvm::sort(accesses, [](const MachineStorageAccess &lhs,
                            const MachineStorageAccess &rhs) {
      return lhs.resource < rhs.resource;
    });

    for (unsigned begin = 0; begin < accesses.size();) {
      unsigned end = begin + 1;
      bool reads = accesses[begin].read;
      bool writes = accesses[begin].write;
      while (end < accesses.size() &&
             accesses[end].resource == accesses[begin].resource) {
        reads |= accesses[end].read;
        writes |= accesses[end].write;
        ++end;
      }

      StorageState &state =
          states.try_emplace(accesses[begin].resource).first->second;
      if (reads) {
        if (state.lastWriter)
          addEdge(graph, *state.lastWriter, index, MachineHazardKind::raw);
        state.readers.push_back(index);
      }
      if (writes) {
        if (state.lastWriter)
          addEdge(graph, *state.lastWriter, index, MachineHazardKind::waw);
        for (unsigned reader : state.readers)
          if (reader != index)
            addEdge(graph, reader, index, MachineHazardKind::war);
        state.readers.clear();
        state.lastWriter = index;
      }
      begin = end;
    }
  }
}

static bool hasPath(const ScheduleGraph &graph, unsigned source,
                    unsigned target) {
  BitVector visited(graph.nodes.size());
  SmallVector<unsigned, 16> pending{source};
  visited.set(source);
  while (!pending.empty()) {
    unsigned node = pending.pop_back_val();
    if (node == target)
      return true;
    for (unsigned edgeIndex : graph.outgoing[node]) {
      unsigned successor = graph.edges[edgeIndex].target;
      if (visited.test(successor))
        continue;
      visited.set(successor);
      pending.push_back(successor);
    }
  }
  return false;
}

static void addExtraEdges(ArrayRef<Operation *> operations,
                          const MachineScheduleModel &model,
                          ScheduleGraph &graph) {
  SmallVector<MachineExtraDependency, 16> dependencies;
  model.getExtraDependencies(operations, dependencies);
  for (const MachineExtraDependency &dependency : dependencies) {
    DenseMap<Operation *, unsigned>::iterator source =
        graph.nodes.find(dependency.source);
    DenseMap<Operation *, unsigned>::iterator target =
        graph.nodes.find(dependency.target);
    if (source == graph.nodes.end() || target == graph.nodes.end())
      continue;
    if (hasPath(graph, target->second, source->second)) {
      // Required storage hazards cannot be dropped; keep this segment intact.
      graph.legal = false;
      return;
    }
    addEdge(graph, source->second, target->second, dependency.kind);
  }
}

static ScheduleGraph buildGraph(ArrayRef<Operation *> operations,
                                const MachineScheduleModel &model) {
  ScheduleGraph graph;
  graph.incoming.resize(operations.size());
  graph.outgoing.resize(operations.size());
  graph.pendingPredecessors.assign(operations.size(), 0);
  for (auto [index, operation] : llvm::enumerate(operations))
    graph.nodes.try_emplace(operation, index);

  // Memory ordering enters only through model-classified SSA token operands.
  for (auto [target, operation] : llvm::enumerate(operations)) {
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition)
        continue;
      DenseMap<Operation *, unsigned>::iterator source =
          graph.nodes.find(definition);
      if (source == graph.nodes.end())
        continue;
      addEdge(graph, source->second, target,
              model.classifyDataDependency(operand));
    }
  }
  addStorageEdges(operations, model, graph);
  addExtraEdges(operations, model, graph);
  return graph;
}

static SmallVector<MachineScheduleDependency, 4>
collectDependencies(unsigned node, const ScheduleGraph &graph,
                    ArrayRef<std::optional<MachineScheduleIssue>> issues,
                    ArrayRef<Operation *> operations) {
  SmallVector<MachineScheduleDependency, 4> dependencies;
  for (unsigned edgeIndex : graph.incoming[node]) {
    const ScheduleEdge &edge = graph.edges[edgeIndex];
    assert(issues[edge.source] && "ready node has an unissued predecessor");
    dependencies.push_back(
        {operations[edge.source], edge.kind, *issues[edge.source]});
  }
  return dependencies;
}

static void markScheduled(unsigned node, const ScheduleGraph &graph,
                          BitVector &ready, BitVector &scheduled,
                          SmallVectorImpl<unsigned> &pending) {
  ready.reset(node);
  scheduled.set(node);
  for (unsigned edgeIndex : graph.outgoing[node]) {
    unsigned successor = graph.edges[edgeIndex].target;
    assert(pending[successor] != 0 && "successor predecessor count underflow");
    --pending[successor];
    if (pending[successor] == 0 && !scheduled.test(successor))
      ready.set(successor);
  }
}

static SmallVector<unsigned, 16> projectPrefixWithNoInstructions(
    unsigned node, const ScheduleGraph &graph,
    const MachineScheduleModel &model, ArrayRef<Operation *> operations,
    ArrayRef<unsigned> order, const BitVector &ready,
    const BitVector &scheduled, ArrayRef<unsigned> pending) {
  SmallVector<unsigned, 16> projectedOrder(order.begin(), order.end());
  BitVector projectedReady = ready;
  BitVector projectedScheduled = scheduled;
  SmallVector<unsigned, 16> projectedPending(pending.begin(), pending.end());
  projectedOrder.push_back(node);
  markScheduled(node, graph, projectedReady, projectedScheduled,
                projectedPending);
  while (true) {
    unsigned selected = operations.size();
    for (int index = projectedReady.find_first(); index >= 0;
         index = projectedReady.find_next(index)) {
      if (!model.isNoInstruction(operations[index]))
        continue;
      selected = index;
      break;
    }
    if (selected == operations.size())
      return projectedOrder;
    projectedOrder.push_back(selected);
    markScheduled(selected, graph, projectedReady, projectedScheduled,
                  projectedPending);
  }
}

static unsigned findFirstUnscheduled(const BitVector &scheduled) {
  for (unsigned index : llvm::seq<unsigned>(scheduled.size()))
    if (!scheduled.test(index))
      return index;
  return scheduled.size();
}

static LogicalResult
commitNode(unsigned node, const ScheduleGraph &graph,
           ArrayRef<Operation *> operations, MachineScheduleState &state,
           SmallVectorImpl<std::optional<MachineScheduleIssue>> &issues,
           SmallVectorImpl<unsigned> &order, BitVector &ready,
           BitVector &scheduled, SmallVectorImpl<unsigned> &pending) {
  SmallVector<MachineScheduleDependency, 4> dependencies =
      collectDependencies(node, graph, issues, operations);
  FailureOr<MachineScheduleIssue> issue =
      state.commitIssue(operations[node], dependencies);
  if (failed(issue))
    return failure();
  issues[node] = *issue;
  order.push_back(node);
  markScheduled(node, graph, ready, scheduled, pending);
  return success();
}

static FailureOr<unsigned> findFirstReadyNoInstruction(
    const BitVector &ready, const ScheduleGraph &graph,
    ArrayRef<std::optional<MachineScheduleIssue>> issues,
    ArrayRef<Operation *> operations, const MachineScheduleState &state) {
  for (int index = ready.find_first(); index >= 0;
       index = ready.find_next(index)) {
    SmallVector<MachineScheduleDependency, 4> dependencies =
        collectDependencies(index, graph, issues, operations);
    FailureOr<MachineScheduleIssue> preview =
        state.previewIssue(operations[index], dependencies);
    if (failed(preview))
      return failure();
    if (!preview->instruction)
      return static_cast<unsigned>(index);
  }
  return ready.size();
}

static LogicalResult drainReadyNoInstructions(
    const ScheduleGraph &graph, ArrayRef<Operation *> operations,
    MachineScheduleState &state,
    SmallVectorImpl<std::optional<MachineScheduleIssue>> &issues,
    SmallVectorImpl<unsigned> &order, BitVector &ready, BitVector &scheduled,
    SmallVectorImpl<unsigned> &pending) {
  while (true) {
    FailureOr<unsigned> selected =
        findFirstReadyNoInstruction(ready, graph, issues, operations, state);
    if (failed(selected))
      return failure();
    if (*selected == operations.size())
      return success();
    if (failed(commitNode(*selected, graph, operations, state, issues, order,
                          ready, scheduled, pending)))
      return failure();
  }
}

static void applyOrder(ArrayRef<Operation *> operations,
                       ArrayRef<unsigned> order) {
  if (llvm::all_of(llvm::enumerate(order),
                   [](auto item) { return item.index() == item.value(); }))
    return;

  Block *block = operations.front()->getBlock();
  Block::iterator insertion = operations.front()->getIterator();
  for (unsigned index : order) {
    Operation *operation = operations[index];
    if (&*insertion != operation)
      operation->moveBefore(block, insertion);
    insertion = std::next(operation->getIterator());
  }
}

static LogicalResult scheduleOperations(ArrayRef<Operation *> operations,
                                        const MachineScheduleModel &model) {
  if (operations.size() < 2)
    return success();

  ScheduleGraph graph = buildGraph(operations, model);
  if (!graph.legal)
    return success();
  std::unique_ptr<MachineScheduleRegionSession> session =
      model.createRegionSession(operations);
  std::unique_ptr<MachineScheduleState> state = model.createState();
  SmallVector<unsigned, 16> pending = graph.pendingPredecessors;
  BitVector ready(operations.size());
  BitVector scheduled(operations.size());
  for (unsigned index : llvm::seq<unsigned>(operations.size()))
    if (pending[index] == 0)
      ready.set(index);

  SmallVector<std::optional<MachineScheduleIssue>, 16> issues(
      operations.size());
  SmallVector<unsigned, 16> order;
  while (order.size() != operations.size()) {
    if (failed(drainReadyNoInstructions(graph, operations, *state, issues,
                                        order, ready, scheduled, pending)))
      return failure();
    if (order.size() == operations.size())
      break;
    unsigned baseline = findFirstUnscheduled(scheduled);
    if (ready.none())
      return operations[baseline]->emitError(
          "machine scheduler dependency cycle");

    unsigned selected = baseline;
    if (!ready.test(baseline)) {
      selected = ready.find_first();
    } else {
      SmallVector<MachineScheduleDependency, 4> baselineDependencies =
          collectDependencies(baseline, graph, issues, operations);
      FailureOr<MachineScheduleIssue> baselinePreview =
          state->previewIssue(operations[baseline], baselineDependencies);
      if (failed(baselinePreview))
        return failure();
      if (baselinePreview->stallCycles != 0) {
        for (int candidate = ready.find_first(); candidate >= 0;
             candidate = ready.find_next(candidate)) {
          if (static_cast<unsigned>(candidate) == baseline)
            continue;
          SmallVector<unsigned, 16> projected = projectPrefixWithNoInstructions(
              candidate, graph, model, operations, order, ready, scheduled,
              pending);
          if (!session->canSchedulePrefix(projected))
            continue;
          SmallVector<MachineScheduleDependency, 4> candidateDependencies =
              collectDependencies(candidate, graph, issues, operations);
          FailureOr<bool> canFill =
              state->canFill(operations[baseline], baselineDependencies,
                             operations[candidate], candidateDependencies);
          if (failed(canFill))
            return failure();
          if (*canFill) {
            selected = candidate;
            break;
          }
        }
      }
    }

    if (failed(commitNode(selected, graph, operations, *state, issues, order,
                          ready, scheduled, pending)))
      return failure();
  }

  if (session->canSchedulePrefix(order))
    applyOrder(operations, order);
  return success();
}

class RegionCollector {
public:
  explicit RegionCollector(const MachineScheduleModel &model) : model(model) {}

  FailureOr<SmallVector<CollectedRegion, 16>> collect(Region &root) {
    if (failed(collectRegion(root)))
      return failure();
    return std::move(regions);
  }

private:
  void flush(Block &block, SmallVectorImpl<Operation *> &operations) {
    if (operations.empty())
      return;
    CollectedRegion &region = regions.emplace_back();
    region.operations.assign(operations.begin(), operations.end());
    region.block = &block;
    operations.clear();
  }

  LogicalResult collectRegion(Region &region) {
    for (Block &block : region)
      if (failed(collectBlock(block)))
        return failure();
    return success();
  }

  LogicalResult collectBlock(Block &block) {
    SmallVector<Operation *, 16> operations;
    for (Operation &operation : block) {
      if (operation.getNumRegions() != 0) {
        flush(block, operations);
        if (!model.isSupportedRegionOperation(&operation))
          return operation.emitError(
              "machine scheduler does not support nested region operation");
        for (Region &nested : operation.getRegions())
          if (failed(collectRegion(nested)))
            return failure();
        continue;
      }
      if (operation.hasTrait<OpTrait::IsTerminator>() ||
          !model.isSchedulable(&operation)) {
        flush(block, operations);
        continue;
      }
      operations.push_back(&operation);
    }
    flush(block, operations);
    return success();
  }

  const MachineScheduleModel &model;
  SmallVector<CollectedRegion, 16> regions;
};

} // namespace

LogicalResult inter::scheduleMachineRegion(Region &region,
                                           const MachineScheduleModel &model) {
  FailureOr<SmallVector<CollectedRegion, 16>> regions =
      RegionCollector(model).collect(region);
  if (failed(regions))
    return failure();
  for (const CollectedRegion &collected : *regions)
    if (failed(scheduleOperations(collected.operations, model)))
      return failure();
  return success();
}
