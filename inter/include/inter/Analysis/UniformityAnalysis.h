// Sparse-forward uniformity analysis (design doc section 7).
//
// Lattice: Bottom < Const < Uniform < Strided(k) < Varying.
//   Const    compile-time known.
//   Uniform  same in every lane of the subgroup.
//   Strided  lane-affine with uniform base and constant stride k; this is the
//            class that picks block vs scatter message forms later.
//   Varying  unknown.
//
// Sources: constants are Const; kernel arguments are Uniform; the
// get_global_id builtin is Strided(1); loads are Varying. Everything else
// propagates. scf.if results propagate through the region-branch interfaces
// for free. Consumed by selection for exec_if vs uniform_if, later by the
// address planner.

#ifndef INTER_ANALYSIS_UNIFORMITYANALYSIS_H
#define INTER_ANALYSIS_UNIFORMITYANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace inter {

enum class UniformityKind { Bottom, Const, Uniform, Strided, Varying };

struct Uniformity {
  UniformityKind kind = UniformityKind::Bottom;
  uint32_t stride = 0;

  static Uniformity bottom() { return {}; }
  static Uniformity constant() { return {UniformityKind::Const, 0}; }
  static Uniformity uniform() { return {UniformityKind::Uniform, 0}; }
  static Uniformity strided(uint32_t s) { return {UniformityKind::Strided, s}; }
  static Uniformity varying() { return {UniformityKind::Varying, 0}; }

  bool isAtMost(UniformityKind k) const { return kind <= k; }

  static Uniformity join(const Uniformity &a, const Uniformity &b) {
    if (a.kind == b.kind) {
      if (a.kind == UniformityKind::Strided && a.stride != b.stride)
        return varying();
      return a;
    }
    return a.kind > b.kind ? a : b;
  }
};

class UniformityLattice : public mlir::dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  mlir::ChangeResult join(const AbstractSparseLattice &rhs) override {
    const Uniformity &r = static_cast<const UniformityLattice &>(rhs).value;
    Uniformity joined = Uniformity::join(value, r);
    if (joined.kind == value.kind && joined.stride == value.stride)
      return mlir::ChangeResult::NoChange;
    value = joined;
    return mlir::ChangeResult::Change;
  }

  const Uniformity &getUniformity() const { return value; }

  void print(mlir::raw_ostream &os) const override {
    static const char *names[] = {"bottom", "const", "uniform", "strided",
                                  "varying"};
    os << "uniformity<" << names[static_cast<int>(value.kind)];
    if (value.kind == UniformityKind::Strided)
      os << " k=" << value.stride;
    os << ">";
  }
  void setUniformity(const Uniformity &u) { value = u; }

  mlir::ChangeResult joinUniformity(const Uniformity &u) {
    Uniformity joined = Uniformity::join(value, u);
    if (joined.kind == value.kind && joined.stride == value.stride)
      return mlir::ChangeResult::NoChange;
    value = joined;
    return mlir::ChangeResult::Change;
  }

private:
  Uniformity value;
};

class UniformityAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<UniformityLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const UniformityLattice *> operands,
                 llvm::ArrayRef<UniformityLattice *> results) override;

  void setToEntryState(UniformityLattice *lattice) override;
};

} // namespace inter

#endif // INTER_ANALYSIS_UNIFORMITYANALYSIS_H
