#ifndef INTER_TRANSFORMS_PASSES_TD
#define INTER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SelectToMachine : Pass<"inter-select-to-machine", "::mlir::ModuleOp"> {
  let summary = "select llvm-dialect kernel to xemachine ops (M1: straight-line)";
  let dependentDialects = [
    "inter::xemachine::XeMachineDialect",
    "mlir::func::FuncDialect",
  ];
}

def NormalizeCf : Pass<"inter-normalize-cf", "::mlir::ModuleOp"> {
  let summary = "llvm.func -> func.func, llvm branches -> cf branches";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect",
  ];
}

#endif // INTER_TRANSFORMS_PASSES_TD
