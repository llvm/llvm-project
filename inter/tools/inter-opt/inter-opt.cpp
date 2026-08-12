#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

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
    declareGeneratedDialect<mlir::memref::MemRefDialect>();
    declareGeneratedDialect<mlir::vector::VectorDialect>();
    declareGeneratedDialect<mlir::gpu::GPUDialect>();
    declareGeneratedDialect<mlir::xegpu::XeGPUDialect>();
    declareGeneratedDialect<mlir::xevm::XeVMDialect>();
  }
};
} // namespace

int main(int argc, char **argv) {
  inter::registerInterPasses();
  mlir::registerTransformsPasses();
  mlir::transform::registerTransformPasses();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createLiftControlFlowToSCFPass();
  });
  mlir::DialectRegistry registry;
  registry.insert<xw::XWDialect, inter::xemachine::XeMachineDialect,
                  mlir::transform::TransformDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                   mlir::cf::ControlFlowDialect, mlir::ub::UBDialect,
                   mlir::LLVM::LLVMDialect, mlir::DLTIDialect,
                   mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                   mlir::gpu::GPUDialect, mlir::xegpu::XeGPUDialect,
                   mlir::xevm::XeVMDialect>();
  registry.addExtensions<InterTransformDialectExtension>();
  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "inter-opt: inter dialect tool\n", registry));
}
