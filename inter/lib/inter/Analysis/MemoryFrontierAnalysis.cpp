#include "inter/Analysis/MemoryFrontierAnalysis.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace inter {

ChangeResult MemoryFrontier::join(const AbstractDenseLattice &rhs) {
  const auto &other = static_cast<const MemoryFrontier &>(rhs);
  ChangeResult changed = ChangeResult::NoChange;
  for (Operation *operation : other.getAccesses())
    changed |= insert(operation);
  return changed;
}

ChangeResult MemoryFrontier::insert(Operation *operation) {
  return accesses.insert(operation) ? ChangeResult::Change
                                    : ChangeResult::NoChange;
}

void MemoryFrontier::print(llvm::raw_ostream &os) const {
  os << "memory-frontier<" << accesses.size() << ">";
}

LogicalResult MemoryFrontierAnalysis::visitOperation(
    Operation *op, const MemoryFrontier &before, MemoryFrontier *after) {
  ChangeResult changed = after->join(before);
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory)
    return success();

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  if (effects.empty())
    return success();

  bool relevant = false;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (isa<MemoryEffects::Read, MemoryEffects::Write, MemoryEffects::Free>(
            effect.getEffect())) {
      relevant = true;
      if (Value location = effect.getValue())
        (void)aliasAnalysis.getModRef(op, location);
    }
  }
  if (relevant)
    changed |= after->insert(op);
  propagateIfChanged(after, changed);
  return success();
}

void MemoryFrontierAnalysis::setToEntryState(MemoryFrontier *lattice) {
  propagateIfChanged(lattice, ChangeResult::NoChange);
}

} // namespace inter.
