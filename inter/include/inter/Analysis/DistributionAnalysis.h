#ifndef INTER_ANALYSIS_DISTRIBUTIONANALYSIS_H
#define INTER_ANALYSIS_DISTRIBUTIONANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "llvm/ADT/SmallVector.h"

namespace inter {

struct Distribution {
  unsigned cardinality = 0;

  static Distribution uninitialized() { return {}; }
  static Distribution bare() { return {1}; }
  static Distribution full(unsigned width) { return {width}; }
  static Distribution join(const Distribution &lhs, const Distribution &rhs);

  bool operator==(const Distribution &rhs) const {
    return cardinality == rhs.cardinality;
  }
  void print(llvm::raw_ostream &os) const;
};

using DistributionLattice = mlir::dataflow::Lattice<Distribution>;

class DistributionAnalysis final
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<
          DistributionLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributionAnalysis)

  DistributionAnalysis(mlir::DataFlowSolver &solver, unsigned simdWidth)
      : SparseForwardDataFlowAnalysis(solver), simdWidth(simdWidth) {}

  llvm::ArrayRef<std::string> getUnknownCauses() const { return unknownCauses; }

  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const DistributionLattice *> operands,
                 llvm::ArrayRef<DistributionLattice *> results) override;

  void visitNonControlFlowArguments(
      mlir::Operation *op, const mlir::RegionSuccessor &successor,
      mlir::ValueRange nonSuccessorInputs,
      llvm::ArrayRef<DistributionLattice *> lattices) override;

  void setToEntryState(DistributionLattice *lattice) override;

private:
  unsigned simdWidth;
  llvm::SmallVector<std::string> unknownCauses;
};

} // namespace inter

#endif // INTER_ANALYSIS_DISTRIBUTIONANALYSIS_H
