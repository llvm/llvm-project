// Op interfaces for xemachine. Semantics land with the passes; defaults here
// are conservative (design doc section 10, "Interfaces").

#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEINTERFACES_TD
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEINTERFACES_TD

include "mlir/IR/OpBase.td"

def XeMachine_SWSBInfoOpInterface : OpInterface<"SWSBInfoOpInterface"> {
  let description = [{
    Software scoreboard classification. The SWSB pass consumes this to
    assign dependency tokens/sync functions. Pipe class 0 means "no scoreboard
    tracking needed".
  }];
  let cppNamespace = "::inter::xemachine";
  let methods = [
    InterfaceMethod<"Dependency pipe class; 0 = untracked.",
                    "unsigned", "getSWSBPipeClass", (ins), /*methodBody=*/"",
                    /*defaultImplementation=*/"return 0;">,
    InterfaceMethod<"True for send-family ops (async completion).",
                    "bool", "isSendMessage", (ins), "",
                    "return false;">,
  ];
}

def XeMachine_RegionLegalityOpInterface : OpInterface<"RegionLegalityOpInterface"> {
  let description = [{
    Operand register-region legality (<V;W,H> strides, alignment). Checked by
    a pass, not by op verifiers (verifiers stay local).
  }];
  let cppNamespace = "::inter::xemachine";
  let methods = [
    InterfaceMethod<"Number of GRFs the result footprint spans at the op's exec size.",
                    "unsigned", "getDstFootprintGRFs", (ins), "",
                    "return 1;">,
  ];
}

def XeMachine_ALUOpInterface : OpInterface<"ALUOpInterface"> {
  let description = [{
    Structural ALU operand metadata used by encoding and register allocation.
  }];
  let cppNamespace = "::inter::xemachine";
  let methods = [
    InterfaceMethod<"Instruction destination element type.",
                    "::mlir::Type", "getInstructionElementType", (ins)>,
    InterfaceMethod<"Source register region, or the canonical region.",
                    "::inter::xemachine::RegionAttr", "getSourceRegion",
                    (ins "unsigned":$index)>,
    InterfaceMethod<"Sets a source register region.", "void",
                    "setSourceRegion",
                    (ins "unsigned":$index,
                         "::inter::xemachine::RegionAttr":$region)>,
    InterfaceMethod<"Explicit source element type override, if present.",
                    "::std::optional<::mlir::Type>",
                    "getExplicitSourceElementType", (ins "unsigned":$index)>,
  ];
}

def XeMachine_InstructionIssueOpInterface : OpInterface<"InstructionIssueOpInterface"> {
  let description = [{
    Timing-oracle hooks. The cost model answers; ops only classify themselves.
  }];
  let cppNamespace = "::inter::xemachine";
  let methods = [
    InterfaceMethod<"Target-independent instruction issue class.",
                    "::inter::xemachine::InstructionIssueClass",
                    "getIssueClass", (ins), "",
                    "return ::inter::xemachine::InstructionIssueClass::none;">,
  ];
}

def XeMachine_FixedPhysicalRegisterDefsOpInterface
    : OpInterface<"FixedPhysicalRegisterDefsOpInterface"> {
  let description = [{
    Ops with architecturally fixed register uses/defs (r0 payload reads,
    implicit ARF uses). Regalloc queries this before touching operands.
  }];
  let cppNamespace = "::inter::xemachine";
  let methods = [
    InterfaceMethod<"True if any operand/result is pinned to a physical register.",
                    "bool", "hasFixedPhysRegs", (ins), "",
                    "return false;">,
  ];
}

def XeMachine_SendDescriptorOpInterface : OpInterface<"SendDescriptorOpInterface"> {
  let description = [{
    Send-message descriptor layout. Exposes descriptor fields so the address
    planner can query operand shapes without hardcoding message tables.
  }];
  let cppNamespace = "::inter::xemachine";
  let methods = [
    InterfaceMethod<"True if the message form accepts lane-varying addresses.",
                    "bool", "supportsScatterAddresses", (ins), "",
                    "return false;">,
  ];
}

def XeMachine_RegisterStorageAliasOpInterface
    : OpInterface<"RegisterStorageAliasOpInterface"> {
  let description = [{
    Zero-cost relative storage constraints exposed to register allocation.
  }];
  let cppNamespace = "::inter::xemachine";
  let methods = [
    InterfaceMethod<
        "Appends register values constrained to share tuple storage.",
        "void", "getRegisterStorageAliases",
        (ins "llvm::SmallVectorImpl<"
             "::inter::xemachine::RegisterStorageAlias> &":$aliases)>,
  ];
}

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEINTERFACES_TD
