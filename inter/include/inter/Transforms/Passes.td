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

def RegAlloc : Pass<"inter-regalloc", "::mlir::func::FuncOp"> {
  let summary = "allocate XeMachine virtual GRFs with iterative relief";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
  let options = [
    Option<"maxIterations", "max-iterations", "unsigned", "32",
           "maximum allocation and relief iterations">,
  ];
}

// Rooted on the module: rewrites the symbol table (llvm.func -> func.func).
def NormalizeCf : Pass<"inter-normalize-cf", "::mlir::ModuleOp"> {
  let summary = "llvm.func -> func.func, llvm branches -> cf branches";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect",
  ];
}

def ConvertCalls : Pass<"inter-convert-calls", "::mlir::func::FuncOp"> {
  let summary = "llvm.call builtins -> xw special ops";
  let dependentDialects = ["xw::XWDialect",
                           "inter::xemachine::XeMachineDialect"];
}

def ConvertMemory : Pass<"inter-convert-memory", "::mlir::func::FuncOp"> {
  let summary = "llvm load/store -> xw token-carrying ops with AA edges";
  let dependentDialects = ["xw::XWDialect",
                           "inter::xemachine::XeMachineDialect"];
}

def InsertSync : Pass<"inter-insert-sync", "::mlir::func::FuncOp"> {
  let summary = "materialize machine sync operations with dense dataflow";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
}

#endif // INTER_TRANSFORMS_PASSES_TD
