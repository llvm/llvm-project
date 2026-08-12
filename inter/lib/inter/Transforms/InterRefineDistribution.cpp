#include "inter/Analysis/DistributionAnalysis.h"
#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "inter/Transforms/Passes.h"

namespace inter {
#define GEN_PASS_DEF_REFINEDISTRIBUTION
#include "inter/Transforms/Passes.h.inc"
} // namespace inter.

using namespace mlir;

namespace {

struct RefineDistribution final
    : inter::impl::RefineDistributionBase<RefineDistribution> {
  using RefineDistributionBase::RefineDistributionBase;

  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      unsigned width = simdWidth;
      if (auto attr = function->getAttrOfType<IntegerAttr>("xw.simd_width"))
        width = attr.getInt();
      if (!width) {
        function.emitOpError("requires a positive xw.simd_width");
        return signalPassFailure();
      }

      DataFlowConfig config;
      config.setInterprocedural(false);
      DataFlowSolver solver(config);
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::SparseConstantPropagation>();
      inter::DistributionAnalysis *analysis =
          solver.load<inter::DistributionAnalysis>(width);
      if (failed(solver.initializeAndRun(function))) {
        function.emitOpError("distribution dataflow failed to converge");
        return signalPassFailure();
      }
      for (StringRef cause : analysis->getUnknownCauses())
        function.emitRemark() << "distribution refinement retained full width: "
                              << cause;

      for (BlockArgument argument : function.getArguments()) {
        const inter::DistributionLattice *lattice =
            solver.lookupState<inter::DistributionLattice>(argument);
        unsigned cardinality = lattice ? lattice->getValue().cardinality : width;
        function.setArgAttr(argument.getArgNumber(), "xw.distribution",
                            IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                             cardinality));
      }

      function.walk([&](Operation *op) {
        if (!op->getNumResults())
          return;
        SmallVector<int32_t> cardinalities;
        cardinalities.reserve(op->getNumResults());
        for (Value result : op->getResults()) {
          const inter::DistributionLattice *lattice =
              solver.lookupState<inter::DistributionLattice>(result);
          unsigned cardinality =
              lattice && lattice->getValue().cardinality
                  ? lattice->getValue().cardinality
                  : width;
          cardinalities.push_back(cardinality);
          if (cardinality != width && result.getType().getDialect().getNamespace() !=
                                          "xw")
            op->emitRemark()
                << "proved cardinality " << cardinality
                << " but retained the current type; XW whole-signature type "
                   "refinement is unavailable";
        }
        op->setAttr("xw.distribution",
                    DenseI32ArrayAttr::get(&getContext(), cardinalities));
      });
    }
  }
};

} // namespace.
