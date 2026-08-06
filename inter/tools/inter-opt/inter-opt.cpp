#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  inter::registerInterPasses();
  mlir::DialectRegistry registry;
  registry.insert<inter::xemachine::XeMachineDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect,
                  mlir::DLTIDialect>();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "inter-opt: inter dialect tool\n", registry));
}
