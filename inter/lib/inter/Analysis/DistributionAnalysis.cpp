#include "inter/Analysis/DistributionAnalysis.h"
#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
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

static Distribution
joinOperands(ArrayRef<const DistributionLattice *> operands) {
  Distribution result = Distribution::uninitialized();
  for (const DistributionLattice *operand : operands)
    result = Distribution::join(result, operand->getValue());
  return result;
}

static Distribution getTypeDistribution(Type type, unsigned simdWidth) {
  if (xw::SimdType simd = dyn_cast<xw::SimdType>(type))
    return {static_cast<unsigned>(simd.getCardinality())};
  if (xw::MaskType mask = dyn_cast<xw::MaskType>(type))
    return {static_cast<unsigned>(mask.getCardinality())};
  if (isa<xw::MemTokenType>(type))
    return Distribution::bare();
  return Distribution::bare();
}

static bool isUniformSource(Operation *op) {
  return isa<arith::ConstantOp, xw::ConstantOp, xw::NullOp,
             xw::LocalMemoryBaseOp, xw::AllocOp, xw::SubgroupIdOp,
             xw::GroupIdOp, xw::GlobalSizeOp, xw::LocalSizeOp, xw::NumGroupsOp,
             xw::LaunchGridSizeOp, xw::LaunchBlockSizeOp>(op);
}

static bool isTokenOperation(Operation *op) {
  return isa<xw::TokenOp, xw::IssueTokenOp, xw::AfterOp, xw::JoinOp,
             xw::StoreOp, xw::Block2DPrefetchOp, xw::Block2DWriteOp,
             xw::BarrierOp, xw::AllocReleaseOp>(op);
}

LogicalResult DistributionAnalysis::visitOperation(
    Operation *op, ArrayRef<const DistributionLattice *> operands,
    ArrayRef<DistributionLattice *> results) {
  Distribution output = joinOperands(operands);

  if (isUniformSource(op) || isTokenOperation(op)) {
    output = Distribution::bare();
  } else if (isa<xw::GlobalIdOp, xw::LocalIdOp, xw::LaneIdOp>(op)) {
    output = Distribution::full(simdWidth);
  } else if (isa<xw::SplatOp, xw::ExpandOp>(op)) {
    output = getTypeDistribution(op->getResult(0).getType(), simdWidth);
  } else if (isa<xw::FreezeOp>(op)) {
    output = operands.front()->getValue();
  } else if (isa<ub::PoisonOp>(op)) {
    output = getTypeDistribution(op->getResult(0).getType(), simdWidth);
  } else if (isa<xw::ReadFirstOp, xw::BallotOp>(op)) {
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
  if (isa<xw::LoadOp, xw::AtomicRMWOp>(op)) {
    assert(results.size() == 2 && "memory operation must have two results");
    propagateIfChanged(results[0],
                       results[0]->join(Distribution::full(simdWidth)));
    propagateIfChanged(results[1], results[1]->join(Distribution::bare()));
    return success();
  }
  if (isa<xw::Block2DReadOp>(op)) {
    assert(results.size() == 2 && "block2D read must have two results");
    propagateIfChanged(results[0], results[0]->join(Distribution::bare()));
    propagateIfChanged(results[1], results[1]->join(Distribution::bare()));
    return success();
  }

  for (auto [result, value] : llvm::zip(results, op->getResults())) {
    Distribution resultDistribution =
        isa<xw::MemTokenType>(value.getType()) ? Distribution::bare() : output;
    propagateIfChanged(result, result->join(resultDistribution));
  }
  return success();
}

void DistributionAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &, ValueRange nonSuccessorInputs,
    ArrayRef<DistributionLattice *> lattices) {
  assert(nonSuccessorInputs.size() == lattices.size() && "size mismatch");
  LoopLikeOpInterface loop = dyn_cast<LoopLikeOpInterface>(op);
  std::optional<SmallVector<Value>> inductionVars =
      loop ? loop.getLoopInductionVars() : std::nullopt;
  std::optional<SmallVector<OpFoldResult>> lowerBounds =
      loop ? loop.getLoopLowerBounds() : std::nullopt;
  std::optional<SmallVector<OpFoldResult>> upperBounds =
      loop ? loop.getLoopUpperBounds() : std::nullopt;
  std::optional<SmallVector<OpFoldResult>> steps =
      loop ? loop.getLoopSteps() : std::nullopt;

  for (auto [input, lattice] : llvm::zip(nonSuccessorInputs, lattices)) {
    Distribution state = Distribution::full(simdWidth);
    if (inductionVars) {
      auto position = llvm::find(*inductionVars, input);
      if (position != inductionVars->end()) {
        unsigned index = std::distance(inductionVars->begin(), position);
        state = Distribution::bare();
        auto joinBound =
            [&](const std::optional<SmallVector<OpFoldResult>> &xs) {
              if (!xs || index >= xs->size())
                return;
              if (Value value = dyn_cast<Value>((*xs)[index])) {
                const DistributionLattice *bound =
                    getLatticeElementFor(getProgramPointBefore(op), value);
                state = Distribution::join(state, bound->getValue());
              }
            };
        joinBound(lowerBounds);
        joinBound(upperBounds);
        joinBound(steps);
      }
    }
    propagateIfChanged(lattice, lattice->join(state));
  }
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
