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

// Rooted on the module: rewrites the symbol table (llvm.func -> func.func).
def NormalizeCf : Pass<"inter-normalize-cf", "::mlir::ModuleOp"> {
  let summary = "import SPIR kernels and normalize LLVM branches to cf";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect",
    "inter::xemachine::XeMachineDialect",
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

def NormalizePointers : Pass<"inter-normalize-pointers", "::mlir::func::FuncOp"> {
  let summary = "normalize LLVM GEPs to bytewise xw.ptradd operations";
  let dependentDialects = ["xw::XWDialect", "mlir::LLVM::LLVMDialect"];
}

def DecomposeWide : Pass<"inter-decompose-wide", "::mlir::func::FuncOp"> {
  let summary = "decompose global pointer offsets into xw wide operations";
  let dependentDialects = ["xw::XWDialect", "mlir::LLVM::LLVMDialect"];
}

def InsertSync : Pass<"inter-insert-sync", "::mlir::func::FuncOp"> {
  let summary = "materialize machine sync operations with dense dataflow";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
}

def PrepareRegAlloc : Pass<"inter-prepare-regalloc", "::mlir::func::FuncOp"> {
  let summary = "normalize XeMachine register aliases before scheduling";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
}

def MachineSchedule : Pass<"inter-machine-schedule", "::mlir::func::FuncOp"> {
  let summary = "schedule XeMachine instructions with greedy gap filling";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
}

def ResourceInfo : Pass<"inter-resource-info", "::mlir::func::FuncOp"> {
  let summary = "derive and validate XeMachine kernel resource metadata";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
}

#endif // INTER_TRANSFORMS_PASSES_TD
