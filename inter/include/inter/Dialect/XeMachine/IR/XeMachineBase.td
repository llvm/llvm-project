// Dialect definition, enums, and attributes for xemachine.
// Design: inter/docs/IntelXeBackendDesign.md section 10.

#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEBASE_TD
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def XeMachineDialect : Dialect {
  let name = "xemachine";
  let summary = "Intel EU machine-level dialect. Xe2 first.";
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
// separate mini-problem from GRF pressure; see design doc section 12.
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
}

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEBASE_TD
