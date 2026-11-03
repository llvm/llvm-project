#ifndef INTER_TRANSFORMS_PASSES_TD
#define INTER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ImportLLVM : Pass<"inter-import-llvm", "::mlir::ModuleOp"> {
  let summary = "import LLVM kernel ABI facts and normalize functions to func/cf";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::cf::ControlFlowDialect"];
  let options = [Option<"simdWidth", "simd-width", "unsigned", "16",
                        "kernel SIMD width (8, 16, or 32)">];
}

def DiscoverCacheControls
    : Pass<"inter-discover-cache-controls", "::mlir::ModuleOp"> {
  let summary = "resolve pointer cache annotations onto memory consumers";
  let dependentDialects = ["xw::XWDialect", "mlir::LLVM::LLVMDialect"];
}

def CanonicalizeBlock2DABI
    : Pass<"inter-canonicalize-block2d-abi", "::mlir::ModuleOp"> {
  let summary = "fold Intel block2D private-pointer ABI shims into XW operations";
  let dependentDialects = ["xw::XWDialect", "mlir::LLVM::LLVMDialect"];
}

def CanonicalizeDpasBuiltin
    : Pass<"inter-canonicalize-dpas-builtin", "::mlir::ModuleOp"> {
  let summary = "fold Intel matrix-mad builtins into XW DPAS operations";
  let dependentDialects = ["xw::XWDialect", "mlir::LLVM::LLVMDialect"];
}

def VerifyStructured : Pass<"inter-verify-structured", "::mlir::ModuleOp"> {
  let summary = "reject unstructured control flow before semantic conversion";
}

def PrepareCountedLoops
    : Pass<"inter-prepare-counted-loops", "::mlir::ModuleOp"> {
  let summary = "normalize imported counted loops for SCF uplifting";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def ConvertLLVMToXW : Pass<"inter-convert-llvm-to-xw", "::mlir::ModuleOp"> {
  let summary = "perform the hard LLVM to XW semantic conversion boundary";
  let dependentDialects = ["xw::XWDialect", "mlir::func::FuncDialect",
                           "mlir::scf::SCFDialect", "mlir::arith::ArithDialect"];
}

def RefineDistribution : Pass<"inter-refine-distribution", "::mlir::ModuleOp"> {
  let summary = "infer compact XW lane distributions and record refinements";
  let dependentDialects = ["xw::XWDialect", "mlir::func::FuncDialect",
                           "mlir::scf::SCFDialect"];
  let options = [Option<"simdWidth", "simd-width", "unsigned", "16",
                        "kernel SIMD width">];
}

def NarrowIntegerRanges
    : Pass<"inter-narrow-integer-ranges", "::mlir::ModuleOp"> {
  let summary = "narrow proven scalar XW integer computations";
  let dependentDialects = ["xw::XWDialect", "mlir::scf::SCFDialect"];
}

def ExpandArithmetic : Pass<"inter-expand-arithmetic", "::mlir::ModuleOp"> {
  let summary = "expand generic XW arithmetic before machine selection";
  let dependentDialects = ["xw::XWDialect"];
}

def InferMemoryTokens : Pass<"inter-infer-memory-tokens", "::mlir::ModuleOp"> {
  let summary = "materialize XW memory dependencies from alias analysis";
  let dependentDialects = ["xw::XWDialect", "mlir::func::FuncDialect",
                           "mlir::scf::SCFDialect"];
}

def SelectToMachine : Pass<"inter-select-to-machine", "::mlir::ModuleOp"> {
  let summary = "select closed XW kernels to XeMachine operations";
  let dependentDialects = [
    "inter::xemachine::XeMachineDialect",
    "mlir::func::FuncDialect",
    "mlir::ub::UBDialect",
  ];
  let options = [
    Option<"chip", "chip", "std::string", "\"bmg\"",
           "Intel GPU target chip">,
    ListOption<"targetFeatures", "target-features", "std::string",
               "Intel GPU target features">,
  ];
}

def CoalesceTuples : Pass<"inter-coalesce-tuples", "::mlir::func::FuncOp"> {
  let summary = "factor common XeMachine tuple updates";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
}

def ReuseBlock2DPayloads : Pass<"inter-reuse-block2d-payloads",
                                "::mlir::func::FuncOp"> {
  let summary = "reuse block2D payloads through immediate send offsets";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
}

def InsertSync : Pass<"inter-insert-sync", "::mlir::func::FuncOp"> {
  let summary = "materialize machine sync operations with dense dataflow";
  let dependentDialects = ["inter::xemachine::XeMachineDialect"];
  let options = [Option<"chip", "chip", "std::string", "\"\"",
                        "Intel GPU target chip for standalone machine IR">];
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
