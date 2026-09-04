#ifndef INTER_ANALYSIS_MEMORYFRONTIERANALYSIS_H
#define INTER_ANALYSIS_MEMORYFRONTIERANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "llvm/ADT/SetVector.h"

namespace inter {

class MemoryFrontier final : public mlir::dataflow::AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryFrontier)
  using AbstractDenseLattice::AbstractDenseLattice;

  mlir::ChangeResult join(const AbstractDenseLattice &rhs) override;
  mlir::ChangeResult insert(mlir::Operation *operation);
  llvm::ArrayRef<mlir::Operation *> getAccesses() const {
    return accesses.getArrayRef();
  }
  void print(llvm::raw_ostream &os) const override;

private:
  llvm::SetVector<mlir::Operation *> accesses;
};

class MemoryFrontierAnalysis final
    : public mlir::dataflow::DenseForwardDataFlowAnalysis<MemoryFrontier> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryFrontierAnalysis)

  MemoryFrontierAnalysis(mlir::DataFlowSolver &solver,
                         mlir::AliasAnalysis &aliasAnalysis)
      : DenseForwardDataFlowAnalysis(solver), aliasAnalysis(aliasAnalysis) {}

  mlir::LogicalResult visitOperation(mlir::Operation *op,
                                     const MemoryFrontier &before,
                                     MemoryFrontier *after) override;
  void setToEntryState(MemoryFrontier *lattice) override;

private:
  mlir::AliasAnalysis &aliasAnalysis;
};

} // namespace inter

#endif // INTER_ANALYSIS_MEMORYFRONTIERANALYSIS_H
