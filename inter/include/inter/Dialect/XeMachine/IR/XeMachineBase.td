// Dialect definition, enums, and attributes for xemachine.
// Design: inter/docs/BackendDialect.md.

#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEBASE_TD
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def XeMachineDialect : Dialect {
  let name = "xemachine";
  let summary = "Intel EU machine-level dialect.";
  let description = [{
    Machine-level IR for Intel Execution Units: register-class-and-width in
    the type, explicit memory tokens, structured control flow as region ops.
    Live from selection through emission; nothing below this dialect is MLIR.
  }];
  let cppNamespace = "::inter::xemachine";
  let useDefaultTypePrinterParser = true;
  let useDefaultAttributePrinterParser = true;
}

// ARF (architecture register file) selector. Allocation of these is a
// separate mini-problem from GRF pressure; see RegisterAllocation.md.
def XeMachine_ARFFile : I32EnumAttr<"ARFFile", "ARF register file", [
    I32EnumAttrCase<"a0",  0>,
    I32EnumAttrCase<"f",   1>,
    I32EnumAttrCase<"acc", 2>,
    I32EnumAttrCase<"mme", 3>,
    I32EnumAttrCase<"sr",  4>,
    I32EnumAttrCase<"cr",  5>,
    I32EnumAttrCase<"n",   6>,
    I32EnumAttrCase<"ip",  7>,
    I32EnumAttrCase<"tdr", 8>,
    I32EnumAttrCase<"tm",  9>,
    I32EnumAttrCase<"fc", 10>,
    I32EnumAttrCase<"dbg", 11>,
  ]> {
  let cppNamespace = "::inter::xemachine";
}

// Compilation target. Only xe2 is populated; the attr leaves room.
def XeMachine_TargetAttr : AttrDef<XeMachineDialect, "Target"> {
  let mnemonic = "target";
  let summary = "Compilation target chip, e.g. #xemachine.target<chip = \"bmg\">";
  let parameters = (ins "mlir::StringAttr":$chip);
  let assemblyFormat = "`<` `chip` `=` $chip `>`";
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "::inter::xemachine::TargetChip":$chip), [{
      switch (chip) {
      case ::inter::xemachine::TargetChip::bmg:
        return $_get($_ctxt, ::mlir::StringAttr::get($_ctxt, "bmg"));
      }
      llvm_unreachable("unknown Intel GPU target chip");
    }]>
  ];
  let genVerifyDecl = 1;
}

def XeMachine_KernelArgKind : I32EnumAttr<"KernelArgKind",
    "kernel argument payload kind", [
      I32EnumAttrCase<"by_value", 0>,
      I32EnumAttrCase<"by_pointer", 1>,
    ]> {
  let cppNamespace = "::inter::xemachine";
}

def XeMachine_KernelArgAttr : AttrDef<XeMachineDialect, "KernelArg"> {
  let mnemonic = "kernel_arg";
  let summary = "self-contained kernel argument ABI descriptor";
  let parameters = (ins "KernelArgKind":$kind,
                          "mlir::StringAttr":$addressSpace,
                          "mlir::StringAttr":$access,
                          "uint64_t":$size, "uint64_t":$alignment,
                          "uint64_t":$offset);
  let assemblyFormat =
      "`<` `kind` `=` $kind `,` `address_space` `=` $addressSpace `,`"
      "`access` `=` $access `,` `size` `=` $size `,` `alignment` `=`"
      "$alignment `,` `offset` `=` $offset `>`";
}

// Register region: <vstride;width,hstride> on sources. Destination regions
// carry only hstride; see DstRegion.
def XeMachine_RegionAttr : AttrDef<XeMachineDialect, "Region"> {
  let mnemonic = "region";
  let summary = "EU register region <v;w,h>";
  let parameters = (ins "uint32_t":$vstride, "uint32_t":$width,
                        "uint32_t":$hstride);
  let assemblyFormat = "`<` $vstride `,` $width `,` $hstride `>`";
}

// Destination region: hstride only (EU dst model). Printed <h>.
def XeMachine_DstRegionAttr : AttrDef<XeMachineDialect, "DstRegion"> {
  let mnemonic = "dstregion";
  let summary = "EU destination region <h>";
  let parameters = (ins "uint32_t":$hstride);
  let assemblyFormat = "`<` $hstride `>`";
}

def XeMachine_SendFn : I32EnumAttr<"SendFn", "shared function", [
    I32EnumAttrCase<"ugm",  0>,
    I32EnumAttrCase<"gtwy", 1>,
    I32EnumAttrCase<"tgm",  2>,
    I32EnumAttrCase<"slm",  3>,
  ]> {
  let cppNamespace = "::inter::xemachine";
}

// Compare condition modifiers (signedness comes from the operand types).
def XeMachine_CondModifier : I32EnumAttr<"CondModifier", "compare modifier", [
    I32EnumAttrCase<"eq", 0>,
    I32EnumAttrCase<"ne", 1>,
    I32EnumAttrCase<"lt", 2>,
    I32EnumAttrCase<"le", 3>,
    I32EnumAttrCase<"gt", 4>,
    I32EnumAttrCase<"ge", 5>,
  ]> {
  let cppNamespace = "::inter::xemachine";
}

def XeMachine_SyncKind : I32EnumAttr<"SyncKind", "sync function", [
    I32EnumAttrCase<"nop",   0>,
    I32EnumAttrCase<"allrd", 1>,
    I32EnumAttrCase<"allwr", 2>,
    I32EnumAttrCase<"bar",   3>,
  ]> {
  let cppNamespace = "::inter::xemachine";
}

def XeMachine_SWSBDistancePipe : I32EnumAttr<"SWSBDistancePipe",
    "SWSB distance pipe", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"all", 1>,
      I32EnumAttrCase<"floating", 2>,
      I32EnumAttrCase<"in_order", 3>,
    ]> {
  let cppNamespace = "::inter::xemachine";
}

def XeMachine_SWSBTokenMode : I32EnumAttr<"SWSBTokenMode",
    "SWSB token operation", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"set", 1>,
      I32EnumAttrCase<"source", 2>,
      I32EnumAttrCase<"destination", 3>,
    ]> {
  let cppNamespace = "::inter::xemachine";
}

def XeMachine_DpasPrecisionF16 : I32EnumAttrCase<"F16", 0, "f16">;
def XeMachine_DpasPrecisionBF16 : I32EnumAttrCase<"BF16", 1, "bf16">;
def XeMachine_DpasPrecision : I32EnumAttr<"DpasPrecision",
    "DPAS source precision",
    [XeMachine_DpasPrecisionF16, XeMachine_DpasPrecisionBF16]> {
  let cppNamespace = "::inter::xemachine";
}

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEBASE_TD
