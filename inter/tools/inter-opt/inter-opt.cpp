#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir {
std::unique_ptr<Pass> createLiftControlFlowToSCFPass();
}

namespace {
class InterTransformDialectExtension
    : public mlir::transform::TransformDialectExtension<
          InterTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InterTransformDialectExtension)

  using Base::Base;

  void init() {
    declareGeneratedDialect<xw::XWDialect>();
    declareGeneratedDialect<inter::xemachine::XeMachineDialect>();
    declareGeneratedDialect<mlir::func::FuncDialect>();
    declareGeneratedDialect<mlir::scf::SCFDialect>();
    declareGeneratedDialect<mlir::arith::ArithDialect>();
    declareGeneratedDialect<mlir::cf::ControlFlowDialect>();
    declareGeneratedDialect<mlir::ub::UBDialect>();
    declareGeneratedDialect<mlir::LLVM::LLVMDialect>();
    declareGeneratedDialect<mlir::DLTIDialect>();
  }
};
} // namespace

int main(int argc, char **argv) {
  inter::registerInterPasses();
  mlir::transform::registerTransformPasses();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createLiftControlFlowToSCFPass();
  });
  mlir::DialectRegistry registry;
  registry.insert<xw::XWDialect, inter::xemachine::XeMachineDialect,
                  mlir::transform::TransformDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::ub::UBDialect,
                  mlir::LLVM::LLVMDialect, mlir::DLTIDialect>();
  registry.addExtensions<InterTransformDialectExtension>();
  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "inter-opt: inter dialect tool\n", registry));
}
