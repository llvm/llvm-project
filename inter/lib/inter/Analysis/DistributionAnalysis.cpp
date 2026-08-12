#include "inter/Analysis/DistributionAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>

using namespace mlir;
using namespace mlir::dataflow;

namespace inter {

Distribution Distribution::join(const Distribution &lhs,
                                const Distribution &rhs) {
  if (!lhs.cardinality)
    return rhs;
  if (!rhs.cardinality)
    return lhs;
  return {static_cast<unsigned>(std::lcm(lhs.cardinality, rhs.cardinality))};
}

void Distribution::print(llvm::raw_ostream &os) const {
  if (!cardinality)
    return void(os << "distribution<uninitialized>");
  os << "distribution<" << cardinality << ">";
}

static Distribution joinOperands(
    ArrayRef<const DistributionLattice *> operands) {
  Distribution result = Distribution::uninitialized();
  for (const DistributionLattice *operand : operands)
    result = Distribution::join(result, operand->getValue());
  return result;
}

LogicalResult DistributionAnalysis::visitOperation(
    Operation *op, ArrayRef<const DistributionLattice *> operands,
    ArrayRef<DistributionLattice *> results) {
  StringRef name = op->getName().getStringRef();
  Distribution output = joinOperands(operands);

  if (isa<arith::ConstantOp>(op) || name == "xw.constant" ||
      name == "xw.null" || name == "xw.local_memory_base" ||
      name == "xw.workgroup_id" || name == "xw.subgroup_id" ||
      name == "xw.subgroup_size" || name == "xw.global_size" ||
      name == "xw.local_size") {
    output = Distribution::bare();
  } else if (name == "xw.global_id" || name == "xw.local_id" ||
             name == "xw.lane_id" || name == "xw.load" ||
             name == "xw.atomic_rmw") {
    output = Distribution::full(simdWidth);
  } else if (name == "xw.token" || name == "xw.issue_token" ||
             name == "xw.after" || name == "xw.join" ||
             name == "xw.store" || name == "xw.barrier") {
    output = Distribution::bare();
  } else if (!op->getNumOperands() && op->getNumResults()) {
    output = Distribution::full(simdWidth);
    unknownCauses.push_back((Twine(op->getName().getStringRef()) +
                             ": result source has unknown lane semantics")
                                .str());
  }

  if (!output.cardinality)
    output = Distribution::full(simdWidth);
  if (output.cardinality > simdWidth || simdWidth % output.cardinality) {
    unknownCauses.push_back((Twine(op->getName().getStringRef()) +
                             ": cardinality does not divide the SIMD width")
                                .str());
    output = Distribution::full(simdWidth);
  }
  for (DistributionLattice *result : results)
    propagateIfChanged(result, result->join(output));
  return success();
}

void DistributionAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &, ValueRange,
    ArrayRef<DistributionLattice *> lattices) {
  Distribution state = Distribution::bare();
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    for (OpOperand &init : loop.getInitsMutable()) {
      const DistributionLattice *lattice =
          getLatticeElementFor(getProgramPointBefore(op), init.get());
      state = Distribution::join(state, lattice->getValue());
    }
  } else if (!isa<RegionBranchOpInterface>(op)) {
    state = Distribution::full(simdWidth);
  }
  for (DistributionLattice *lattice : lattices)
    propagateIfChanged(lattice, lattice->join(state));
}

void DistributionAnalysis::setToEntryState(DistributionLattice *lattice) {
  Value value = lattice->getAnchor();
  Distribution state = Distribution::full(simdWidth);
  if (BlockArgument argument = dyn_cast<BlockArgument>(value)) {
    Block *block = argument.getOwner();
    if (block->isEntryBlock() && isa<func::FuncOp>(block->getParentOp()))
      state = Distribution::bare();
  }
  propagateIfChanged(lattice, lattice->join(state));
}

} // namespace inter.
