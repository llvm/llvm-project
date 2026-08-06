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

#endif // INTER_TRANSFORMS_PASSES_TD
