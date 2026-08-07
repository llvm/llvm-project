// xemachine ops. One base class injects the interface set; per-op metadata
// lands as TableGen fields when the passes need them (design doc section 10).

#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEOPS_TD
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEOPS_TD

include "inter/Dialect/XeMachine/IR/XeMachineTypes.td"
include "inter/Dialect/XeMachine/IR/XeMachineInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class XeMachine_Op<string mnemonic, list<Trait> traits = []> :
    Op<XeMachineDialect, mnemonic, traits # [
      DeclareOpInterfaceMethods<XeMachine_SWSBInfoOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_RegionLegalityOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_InstructionIssueOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_FixedPhysicalRegisterDefsOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_SendDescriptorOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_RegisterStorageAliasOpInterface>,
    ]> {
  let cppNamespace = "::inter::xemachine";
}

def XeMachine_RegOrImm : AnyTypeOf<[XeMachine_RegType, XeMachine_ImmType,
                                    XeMachine_ARFType], "reg/arf/imm">;

// SSA immediate. Element type rides on the op: immediates are untyped in the
// encoding, typed by the consuming instruction.
def XeMachine_ImmOp : XeMachine_Op<"imm", [NoMemoryEffect]> {
  let summary = "materialize an immediate";
  let arguments = (ins I64Attr:$value, TypeAttr:$elemType);
  let results = (outs XeMachine_ImmType:$result);
  let assemblyFormat = "$value attr-dict `:` $elemType";
}

// Region annotations are operand-register-region strings ("<1;1,0>" etc).
// Absent means canonical: dst <1>, src <1;1,0>. Region legality is checked
// by a pass via RegionLegalityOpInterface, not by op verifiers.
class XeMachine_ALUOp<string mnemonic, list<Trait> traits = []> :
    XeMachine_Op<mnemonic, traits # [NoMemoryEffect]> {
  let cppNamespace = "::inter::xemachine";
}

def XeMachine_MovOp : XeMachine_ALUOp<"mov"> {
  let summary = "register/immediate move";
  let arguments = (ins TypeAttr:$elemType,
                       DefaultValuedAttr<I32Attr, "16">:$execSize,
                       OptionalAttr<XeMachine_DstRegionAttr>:$dstRegion,
                       OptionalAttr<XeMachine_RegionAttr>:$src0Region,
                       OptionalAttr<I32Attr>:$dstSub,
                       OptionalAttr<I32Attr>:$src0Sub,
                       OptionalAttr<TypeAttr>:$src0Type,
                       UnitAttr:$noMask,
                       DefaultValuedAttr<I32Attr, "0">:$maskOffset,
                       XeMachine_RegOrImm:$src);
  let results = (outs XeMachine_RegType:$dst);
  let assemblyFormat =
      "$src attr-dict `:` `(` qualified(type($src)) `,` $elemType `)` `->` qualified(type($dst))";
}

class XeMachine_BinOp<string mnemonic> : XeMachine_ALUOp<mnemonic> {
  let arguments = (ins TypeAttr:$elemType,
                       DefaultValuedAttr<I32Attr, "16">:$execSize,
                       OptionalAttr<XeMachine_DstRegionAttr>:$dstRegion,
                       OptionalAttr<XeMachine_RegionAttr>:$src0Region,
                       OptionalAttr<XeMachine_RegionAttr>:$src1Region,
                       OptionalAttr<I32Attr>:$dstSub,
                       OptionalAttr<I32Attr>:$src0Sub,
                       OptionalAttr<I32Attr>:$src1Sub,
                       OptionalAttr<TypeAttr>:$src0Type,
                       OptionalAttr<TypeAttr>:$src1Type,
                       UnitAttr:$noMask,
                       DefaultValuedAttr<I32Attr, "0">:$maskOffset,
                       XeMachine_RegOrImm:$src0, XeMachine_RegOrImm:$src1);
  let results = (outs XeMachine_RegType:$dst);
  let assemblyFormat =
      "$src0 `,` $src1 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` $elemType `)` `->` qualified(type($dst))";
}

def XeMachine_AddOp : XeMachine_BinOp<"add"> { let summary = "add"; }
def XeMachine_ShlOp : XeMachine_BinOp<"shl"> { let summary = "shift left"; }
def XeMachine_AndOp : XeMachine_BinOp<"and"> { let summary = "bitwise and"; }
def XeMachine_OrOp  : XeMachine_BinOp<"or">  { let summary = "bitwise or"; }

def XeMachine_Add3Op : XeMachine_ALUOp<"add3"> {
  let summary = "three-input add";
  let arguments = (ins TypeAttr:$elemType,
                       DefaultValuedAttr<I32Attr, "16">:$execSize,
                       OptionalAttr<XeMachine_DstRegionAttr>:$dstRegion,
                       OptionalAttr<XeMachine_RegionAttr>:$src0Region,
                       OptionalAttr<XeMachine_RegionAttr>:$src1Region,
                       OptionalAttr<XeMachine_RegionAttr>:$src2Region,
                       OptionalAttr<I32Attr>:$dstSub,
                       OptionalAttr<I32Attr>:$src0Sub,
                       OptionalAttr<I32Attr>:$src1Sub,
                       OptionalAttr<I32Attr>:$src2Sub,
                       OptionalAttr<TypeAttr>:$src0Type,
                       OptionalAttr<TypeAttr>:$src1Type,
                       OptionalAttr<TypeAttr>:$src2Type,
                       UnitAttr:$noMask,
                       DefaultValuedAttr<I32Attr, "0">:$maskOffset,
                       XeMachine_RegOrImm:$src0, XeMachine_RegOrImm:$src1,
                       XeMachine_RegOrImm:$src2);
  let results = (outs XeMachine_RegType:$dst);
  let assemblyFormat =
      "$src0 `,` $src1 `,` $src2 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `,` $elemType `)` `->` qualified(type($dst))";
}

// Shared-function message. Null destination is a reg with widthDwords = 0
// (stores, EOT). Descriptor immediates stay raw until the send-descriptor
// subsystem lands.
def XeMachine_SendOp : XeMachine_Op<"send",
    [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "shared-function message";
  let arguments = (ins XeMachine_SendFn:$fn, I32Attr:$sfid, I32Attr:$desc,
                       I32Attr:$exdesc,
                       DefaultValuedAttr<I32Attr, "1">:$execSize,
                       UnitAttr:$noMask,
                       UnitAttr:$eot,
                       XeMachine_RegType:$addrPayload,
                       Optional<XeMachine_RegType>:$dataPayload,
                       Optional<XeMachine_MemTokenType>:$dependency,
                       OptionalAttr<I64Attr>:$swsb);
  let results = (outs XeMachine_RegType:$dst, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$fn $addrPayload (`data` $dataPayload^)? (`dep` $dependency^)? attr-dict `:` `(` qualified(type($addrPayload)) (`,` qualified(type($dataPayload))^)? `)` `->` `(` qualified(type($dst)) `,` qualified(type($token)) `)`";
}

// Scoreboard/fence synchronization. kind: "nop", "allrd", "allwr", ...
def XeMachine_SyncOp : XeMachine_Op<"sync"> {
  let summary = "software scoreboard synchronization";
  let arguments = (ins XeMachine_SyncKind:$kind);
  let assemblyFormat = "$kind attr-dict";
}

// Structured control flow. exec_if = mask-stack push/pop; uniform_if =
// scalar-condition branch. The emitter sees control flow only as these
// region ops.
class XeMachine_IfOp<string mnemonic> :
    XeMachine_Op<mnemonic, [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                                      ["getSuccessorInputs"]>]> {
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
  let cppNamespace = "::inter::xemachine";
}

def XeMachine_ExecIfOp : XeMachine_IfOp<"exec_if"> {
  let summary = "divergent branch: saves/installs/restores the exec mask";
  let arguments = (ins AnyType:$cond);
  let assemblyFormat =
      "$cond `:` qualified(type($cond)) $thenRegion (`otherwise` $elseRegion^)? attr-dict (`->` qualified(type($results))^)?";
}

def XeMachine_UniformIfOp : XeMachine_IfOp<"uniform_if"> {
  let summary = "uniform branch: condition is subgroup-uniform";
  let arguments = (ins AnyType:$cond);
  let assemblyFormat =
      "$cond `:` qualified(type($cond)) $thenRegion (`otherwise` $elseRegion^)? attr-dict (`->` qualified(type($results))^)?";
}

def XeMachine_UniformLoopOp : XeMachine_Op<"uniform_loop",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                               ["getEntrySuccessorOperands",
                                "getSuccessorInputs"]>]> {
  let summary = "structured loop with explicit carried values";
  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
  let assemblyFormat =
      "`(` $inits `)` $body attr-dict `:` functional-type($inits, $results)";
}

def XeMachine_YieldOp : XeMachine_Op<"yield",
    [Pure, Terminator, ReturnLike,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "region yield";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` qualified(type($values)))?";
}

def XeMachine_ContinueIfOp : XeMachine_Op<"continue_if",
    [Pure, Terminator, ReturnLike,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                               ["getMutableSuccessorOperands"]>]> {
  let summary = "loop back-edge: continue while cond holds, else exit";
  let arguments = (ins AnyType:$cond, Variadic<AnyType>:$carried);
  let assemblyFormat =
      "$cond `:` qualified(type($cond)) attr-dict (`(` $carried^ `:` qualified(type($carried)) `)`)?";
}

def XeMachine_MulOp : XeMachine_ALUOp<"mul"> {
  let summary = "multiply, accumulator destination";
  let arguments = (ins TypeAttr:$elemType,
                       DefaultValuedAttr<I32Attr, "16">:$execSize,
                       OptionalAttr<XeMachine_DstRegionAttr>:$dstRegion,
                       OptionalAttr<XeMachine_RegionAttr>:$src0Region,
                       OptionalAttr<XeMachine_RegionAttr>:$src1Region,
                       OptionalAttr<I32Attr>:$dstSub,
                       OptionalAttr<I32Attr>:$src0Sub,
                       OptionalAttr<I32Attr>:$src1Sub,
                       UnitAttr:$noMask,
                       DefaultValuedAttr<I32Attr, "0">:$maskOffset,
                       XeMachine_RegOrImm:$src0, XeMachine_RegOrImm:$src1);
  let results = (outs XeMachine_ARFType:$dst);
  let assemblyFormat =
      "$src0 `,` $src1 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` $elemType `)` `->` qualified(type($dst))";
}

// Compare: writes a flag register, no GRF destination.
// cmp (exec|M0) (gt)f0.0 null<1>:ud src0 src1
def XeMachine_CmpOp : XeMachine_Op<"cmp", [NoMemoryEffect]> {
  let summary = "compare into a flag register";
  let arguments = (ins XeMachine_CondModifier:$cond,
                       TypeAttr:$elemType,
                       DefaultValuedAttr<I32Attr, "16">:$execSize,
                       OptionalAttr<XeMachine_RegionAttr>:$src0Region,
                       OptionalAttr<XeMachine_RegionAttr>:$src1Region,
                       OptionalAttr<I32Attr>:$src0Sub,
                       OptionalAttr<I32Attr>:$src1Sub,
                       OptionalAttr<TypeAttr>:$src0Type,
                       OptionalAttr<TypeAttr>:$src1Type,
                       XeMachine_RegOrImm:$src0, XeMachine_RegOrImm:$src1);
  let results = (outs XeMachine_ARFType:$flag);
  let assemblyFormat =
      "$cond $src0 `,` $src1 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` $elemType `)` `->` qualified(type($flag))";
}

// Architectural register reference (r0 header, r1 inline args). The result
// carries a fixed physical base; regalloc must not move it.
def XeMachine_ArchRegOp : XeMachine_Op<"archreg", [NoMemoryEffect]> {
  let summary = "reference an architectural register (r0/r1)";
  let arguments = (ins I32Attr:$index);
  let results = (outs XeMachine_RegType:$result);
  let assemblyFormat = "$index attr-dict `:` qualified(type($result))";
}

// ARF register reference (acc, flags, ...).
def XeMachine_ArfRegOp : XeMachine_Op<"arfreg", [NoMemoryEffect]> {
  let summary = "reference an ARF register";
  let arguments = (ins XeMachine_ARFFile:$file, I32Attr:$index);
  let results = (outs XeMachine_ARFType:$result);
  let assemblyFormat = "$file `,` $index attr-dict `:` qualified(type($result))";
}

// Token algebra. token: entry source. token_join: merge completions.
// after: issue-order sequencing without a completion promise.
def XeMachine_TokenOp : XeMachine_Op<"token", [NoMemoryEffect]> {
  let summary = "initial memory-ordering token";
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "attr-dict";
}

def XeMachine_TokenJoinOp : XeMachine_Op<"token_join", [NoMemoryEffect]> {
  let summary = "merge memory-ordering tokens";
  let arguments = (ins Variadic<XeMachine_MemTokenType>:$dependencies);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "$dependencies attr-dict `:` qualified(type($dependencies))";
}

def XeMachine_AfterOp : XeMachine_Op<"after", [NoMemoryEffect]> {
  let summary = "issue-order sequencing of memory ops";
  let arguments = (ins Variadic<XeMachine_MemTokenType>:$dependencies);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "$dependencies attr-dict `:` qualified(type($dependencies))";
}

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEOPS_TD
