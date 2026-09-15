// xemachine ops. One base class injects the interface set; per-op metadata
// lands as TableGen fields when the passes need them (see BackendDialect.md).

#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEOPS_TD
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEOPS_TD

include "inter/Dialect/XeMachine/IR/XeMachineTypes.td"
include "inter/Dialect/XeMachine/IR/XeMachineInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class XeMachine_Op<string mnemonic, list<Trait> traits = [],
                   string issueClass = "none",
                   string instructionKind = "none"> :
    Op<XeMachineDialect, mnemonic, traits # [
      DeclareOpInterfaceMethods<XeMachine_RegionLegalityOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_InstructionIssueOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_FixedPhysicalRegisterDefsOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_SendDescriptorOpInterface>,
  ]> {
  let cppNamespace = "::inter::xemachine";
  let extraClassDeclaration = [{
    ::inter::xemachine::InstructionIssueClass getIssueClass() {
      return ::inter::xemachine::InstructionIssueClass::}] # issueClass # [{;
    }
    ::inter::xemachine::MachineInstructionKind getInstructionKind() {
      return ::inter::xemachine::MachineInstructionKind::}] # instructionKind # [{;
    }
  }];
}

def XeMachine_RegOrImm : AnyTypeOf<[XeMachine_RegType, XeMachine_ImmType,
                                    XeMachine_ARFType], "reg/arf/imm">;
def XeMachine_RegOrARF : AnyTypeOf<[XeMachine_RegType, XeMachine_ARFType],
                                    "reg/arf">;

def XeMachine_NoMachineInst
    : NativeOpTrait<"xemachine::NoMachineInst">;
def XeMachine_NoAsmEmission
    : NativeOpTrait<"xemachine::NoAsmEmission">;
def XeMachine_CompletionFree
    : NativeOpTrait<"xemachine::CompletionFree">;
def XeMachine_FullScoreboardDrain
    : NativeOpTrait<"xemachine::FullScoreboardDrain">;
def XeMachine_Rematerializable
    : NativeOpTrait<"xemachine::Rematerializable">;

// SSA immediate. Element type rides on the op: immediates are untyped in the
// encoding, typed by the consuming instruction.
def XeMachine_ImmOp : XeMachine_Op<"imm", [Pure,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission]> {
  let summary = "materialize an immediate";
  let arguments = (ins I64Attr:$value, TypeAttr:$elemType);
  let results = (outs XeMachine_ImmType:$result);
  let assemblyFormat = "$value attr-dict `:` $elemType";
}

// Region annotations are operand-register-region strings ("<1;1,0>" etc).
// Absent means canonical: dst <1>, src <1;1,0>. Region legality is checked
// by a pass via RegionLegalityOpInterface, not by op verifiers.
class XeMachine_ALUOp<string mnemonic, list<Trait> traits = [],
                       string issueClass = "moveOrLogic",
                       string instructionKind = "mov"> :
    XeMachine_Op<mnemonic, traits # [NoMemoryEffect,
      DeclareOpInterfaceMethods<XeMachine_ALUOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_SWSBInfoOpInterface>], issueClass,
      instructionKind> {
  let cppNamespace = "::inter::xemachine";
}

class XeMachine_SWSBFields {
  dag swsbFields = (ins
      DefaultValuedAttr<XeMachine_SWSBDistancePipe, "SWSBDistancePipe::none">:$swsbPipe,
      DefaultValuedAttr<I32Attr, "-1">:$swsbDistance,
      DefaultValuedAttr<I32Attr, "-1">:$swsbToken,
      DefaultValuedAttr<XeMachine_SWSBTokenMode, "SWSBTokenMode::none">:$swsbTokenMode);
}

def XeMachine_MovOp : XeMachine_ALUOp<"mov", [
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    XeMachine_Rematerializable]>, XeMachine_SWSBFields {
  let summary = "register/immediate move";
  let arguments = !con((ins TypeAttr:$elemType,
                       DefaultValuedAttr<I32Attr, "16">:$execSize,
                       OptionalAttr<XeMachine_DstRegionAttr>:$dstRegion,
                       OptionalAttr<XeMachine_RegionAttr>:$src0Region,
                        OptionalAttr<I32Attr>:$dstSub,
                        OptionalAttr<I32Attr>:$src0Sub,
                        OptionalAttr<TypeAttr>:$src0Type,
                        UnitAttr:$noMask,
                       DefaultValuedAttr<I32Attr, "0">:$maskOffset,
                        XeMachine_RegOrImm:$src), swsbFields);
  let results = (outs XeMachine_RegOrARF:$dst);
  let assemblyFormat =
      "$src attr-dict `:` `(` qualified(type($src)) `,` $elemType `)` `->` qualified(type($dst))";
}

class XeMachine_BinOp<string mnemonic, string instructionKind,
                       string issueClass = "arithmetic",
                       list<Trait> traits = []> :
    XeMachine_ALUOp<mnemonic, traits, issueClass, instructionKind>,
    XeMachine_SWSBFields {
  let arguments = !con((ins TypeAttr:$elemType,
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
                       XeMachine_RegOrImm:$src0, XeMachine_RegOrImm:$src1), swsbFields);
  let results = (outs XeMachine_RegOrARF:$dst);
  let assemblyFormat =
      "$src0 `,` $src1 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` $elemType `)` `->` qualified(type($dst))";
}

def XeMachine_AddOp : XeMachine_BinOp<"add", "add", "arithmetic",
    [XeMachine_Rematerializable]> { let summary = "add"; }
def XeMachine_ShlOp : XeMachine_BinOp<"shl", "shl", "arithmetic",
    [XeMachine_Rematerializable]> { let summary = "shift left"; }
def XeMachine_ShrOp : XeMachine_BinOp<"shr", "shr"> {
  let summary = "logical shift right";
}
def XeMachine_AndOp : XeMachine_BinOp<"and", "and_", "moveOrLogic",
    [XeMachine_Rematerializable]> {
  let summary = "bitwise and";
}
def XeMachine_OrOp : XeMachine_BinOp<"or", "or_", "moveOrLogic",
    [XeMachine_Rematerializable]> {
  let summary = "bitwise or";
}

// EU sub is add with a negated first source; the emitter prints the add
// mnemonic with a negated src0.
def XeMachine_SubOp : XeMachine_BinOp<"sub", "sub", "arithmetic",
    [XeMachine_Rematerializable]> {
  let summary = "sub: add(-s0, s1)";
}

def XeMachine_Add3Op : XeMachine_ALUOp<"add3", [XeMachine_Rematerializable],
    "arithmetic", "add3">,
    XeMachine_SWSBFields {
  let summary = "three-input add";
  let arguments = !con((ins TypeAttr:$elemType,
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
                       XeMachine_RegOrImm:$src2), swsbFields);
  let results = (outs XeMachine_RegType:$dst);
  let assemblyFormat =
      "$src0 `,` $src1 `,` $src2 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `,` $elemType `)` `->` qualified(type($dst))";
}

def XeMachine_CselOp : XeMachine_ALUOp<"csel", [], "arithmetic", "csel">,
    XeMachine_SWSBFields {
  let summary = "conditional select and compare into a flag register";
  let arguments = !con((ins XeMachine_CondModifier:$cond,
                       TypeAttr:$elemType,
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
                       UnitAttr:$signedInt,
                       UnitAttr:$noMask,
                       DefaultValuedAttr<I32Attr, "0">:$maskOffset,
                       XeMachine_RegOrImm:$src0, XeMachine_RegOrImm:$src1,
                       XeMachine_RegOrImm:$src2), swsbFields);
  let results = (outs XeMachine_RegType:$dst, XeMachine_ARFType:$flag);
  let assemblyFormat =
      "$cond $src0 `,` $src1 `,` $src2 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `,` $elemType `)` `->` `(` qualified(type($dst)) `,` qualified(type($flag)) `)`";
}

// Shared-function message. Null destination is a reg with widthDwords = 0
// (stores, EOT). Descriptor immediates stay raw until the send-descriptor
// subsystem lands.
def XeMachine_SendOp : XeMachine_Op<"send",
    [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<XeMachine_SWSBInfoOpInterface>,
     DeclareOpInterfaceMethods<XeMachine_AsyncScoreboardOpInterface>], "send",
    "send">, XeMachine_SWSBFields {
  let summary = "shared-function message";
  let arguments = !con((ins XeMachine_SendFn:$fn, I32Attr:$sfid, I32Attr:$desc,
                       I32Attr:$exdesc,
                       DefaultValuedAttr<I32Attr, "1">:$execSize,
                       UnitAttr:$noMask,
                       UnitAttr:$eot,
                       XeMachine_RegType:$addrPayload,
                       Optional<XeMachine_RegType>:$dataPayload,
                       Optional<XeMachine_ARFType>:$exdescReg,
                       Optional<XeMachine_MemTokenType>:$dependency), swsbFields);
  let results = (outs XeMachine_RegType:$dst, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$fn $addrPayload (`data` $dataPayload^)? (`exdesc` $exdescReg^ `:` qualified(type($exdescReg)))? (`dep` $dependency^)? attr-dict `:` `(` qualified(type($addrPayload)) (`,` qualified(type($dataPayload))^)? `)` `->` `(` qualified(type($dst)) `,` qualified(type($token)) `)`";
}

// Scoreboard/fence synchronization. kind: "nop", "allrd", "allwr", "bar".
// Token-connected like everything else that orders memory.
def XeMachine_SyncOp : XeMachine_Op<"sync",
    [DeclareOpInterfaceMethods<XeMachine_SWSBInfoOpInterface>], "sync", "sync">,
    XeMachine_SWSBFields {
  let summary = "software scoreboard synchronization";
  let arguments = !con((ins XeMachine_SyncKind:$kind,
                         Optional<XeMachine_MemTokenType>:$dependency,
                         DefaultValuedAttr<I32Attr, "0">:$sbidMask), swsbFields);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$kind (`dep` $dependency^)? attr-dict `:` qualified(type($token))";
  let hasVerifier = 1;
}

// Named message ops: the convenient aliases over raw send. Each carries its
// message semantics; the emitter owns the descriptor encodings. Memory
// ordering between them is explicit token edges (see Synchronization.md).
class XeMachine_MsgOp<string mnemonic, list<Trait> traits = []> :
    XeMachine_Op<mnemonic, traits # [
      DeclareOpInterfaceMethods<XeMachine_SWSBInfoOpInterface>,
      DeclareOpInterfaceMethods<XeMachine_AsyncScoreboardOpInterface>], "send",
      "send">, XeMachine_SWSBFields {
  let cppNamespace = "::inter::xemachine";
}

// A64 scattered dword load: load.ugm.d32.a64.
def XeMachine_LoadA64Op : XeMachine_MsgOp<"load_a64", [MemoryEffects<[MemRead]>]> {
  let arguments = !con((ins XeMachine_RegType:$addrPayload,
                       Optional<XeMachine_MemTokenType>:$dependency,
                       DefaultValuedAttr<I32Attr, "32">:$execSize), swsbFields);
  let results = (outs XeMachine_RegType:$dst, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$addrPayload (`dep` $dependency^)? attr-dict `:` qualified(type($addrPayload)) `->` `(` qualified(type($dst)) `,` qualified(type($token)) `)`";
  let hasVerifier = 1;
}

// A64 scattered dword store: store.ugm.d32.a64.
def XeMachine_StoreA64Op : XeMachine_MsgOp<"store_a64", [MemoryEffects<[MemWrite]>]> {
  let arguments = !con((ins XeMachine_RegType:$addrPayload,
                       XeMachine_RegType:$dataPayload,
                       Optional<XeMachine_MemTokenType>:$dependency,
                       DefaultValuedAttr<I32Attr, "32">:$execSize), swsbFields);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$addrPayload `data` $dataPayload (`dep` $dependency^)? attr-dict `:` `(` qualified(type($addrPayload)) `,` qualified(type($dataPayload)) `)` `->` qualified(type($token))";
  let hasVerifier = 1;
}

// SLM dword load/store: load/store.slm.d32.a32.
def XeMachine_LoadSLMOp : XeMachine_MsgOp<"load_slm", [MemoryEffects<[MemRead]>]> {
  let arguments = !con((ins XeMachine_RegType:$addrPayload,
                       Optional<XeMachine_MemTokenType>:$dependency,
                       DefaultValuedAttr<I32Attr, "32">:$execSize), swsbFields);
  let results = (outs XeMachine_RegType:$dst, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$addrPayload (`dep` $dependency^)? attr-dict `:` qualified(type($addrPayload)) `->` `(` qualified(type($dst)) `,` qualified(type($token)) `)`";
}

def XeMachine_StoreSLMOp : XeMachine_MsgOp<"store_slm", [MemoryEffects<[MemWrite]>]> {
  let arguments = !con((ins XeMachine_RegType:$addrPayload,
                       XeMachine_RegType:$dataPayload,
                       Optional<XeMachine_MemTokenType>:$dependency,
                       DefaultValuedAttr<I32Attr, "32">:$execSize), swsbFields);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$addrPayload `data` $dataPayload (`dep` $dependency^)? attr-dict `:` `(` qualified(type($addrPayload)) `,` qualified(type($dataPayload)) `)` `->` qualified(type($token))";
}

// A64 atomic iadd: atomic_iadd.ugm.d32.a64.
def XeMachine_AtomicIAddA64Op : XeMachine_MsgOp<"atomic_iadd_a64",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = !con((ins XeMachine_RegType:$addrPayload,
                       XeMachine_RegType:$dataPayload,
                       Optional<XeMachine_MemTokenType>:$dependency,
                       DefaultValuedAttr<I32Attr, "32">:$execSize), swsbFields);
  let results = (outs XeMachine_RegType:$dst, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$addrPayload `data` $dataPayload (`dep` $dependency^)? attr-dict `:` `(` qualified(type($addrPayload)) `,` qualified(type($dataPayload)) `)` `->` `(` qualified(type($dst)) `,` qualified(type($token)) `)`";
  let hasVerifier = 1;
}

// a32 block (transpose) load via bti[255]; used by the payload prologue.
// words selects the message width (8/16/32 dwords).
def XeMachine_LoadBlockA32Op : XeMachine_MsgOp<"load_block_a32",
    [MemoryEffects<[MemRead]>]> {
  let arguments = !con((ins XeMachine_RegType:$addrPayload,
                       Optional<XeMachine_MemTokenType>:$dependency,
                       I32Attr:$words), swsbFields);
  let results = (outs XeMachine_RegType:$dst, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$addrPayload (`dep` $dependency^)? attr-dict `:` qualified(type($addrPayload)) `->` `(` qualified(type($dst)) `,` qualified(type($token)) `)`";
}

// SLM group fence: fence.slm.none.group. The readback register must be
// drained (fence_await) before signaling.
def XeMachine_FenceSLMOp : XeMachine_MsgOp<"fence_slm",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = !con((ins XeMachine_RegType:$addrPayload,
                       Optional<XeMachine_MemTokenType>:$dependency), swsbFields);
  let results = (outs XeMachine_RegType:$readback, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$addrPayload (`dep` $dependency^)? attr-dict `:` qualified(type($addrPayload)) `->` `(` qualified(type($readback)) `,` qualified(type($token)) `)`";
}

// Fence drain: waits for the fence readback. Prints as a null-dst mov.
def XeMachine_FenceAwaitOp : XeMachine_Op<"fence_await",
    [DeclareOpInterfaceMethods<XeMachine_SWSBInfoOpInterface>], "moveOrLogic",
    "mov">, XeMachine_SWSBFields {
  let arguments = !con((ins XeMachine_RegType:$readback,
                       Optional<XeMachine_MemTokenType>:$dependency), swsbFields);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$readback (`dep` $dependency^)? attr-dict `:` qualified(type($readback)) `->` qualified(type($token))";
}

// Barrier signal to the thread spawner: gtwy message.
def XeMachine_BarrierSignalOp : XeMachine_MsgOp<"barrier_signal",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = !con((ins XeMachine_RegType:$payload,
                       Optional<XeMachine_MemTokenType>:$dependency), swsbFields);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$payload (`dep` $dependency^)? attr-dict `:` qualified(type($payload)) `->` qualified(type($token))";
}

// End of thread.
def XeMachine_EotOp : XeMachine_MsgOp<"eot",
    [XeMachine_FullScoreboardDrain]> {
  let arguments = !con((ins XeMachine_RegType:$payload,
                       Optional<XeMachine_MemTokenType>:$dependency), swsbFields);
  let assemblyFormat =
      "$payload (`dep` $dependency^)? attr-dict `:` qualified(type($payload))";
}

// Software setup skipped when the walker provides the architectural payload.
// The region has no SSA interface because both entry paths communicate only
// through the fixed architectural registers.
def XeMachine_PayloadPrologueOp : XeMachine_Op<"payload_prologue",
    [SingleBlock, NoRegionArguments, RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface,
                               ["getSuccessorInputs"]>]> {
  let summary = "software setup for the alternate kernel payload entry";
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$body attr-dict";
}

def XeMachine_PayloadPrologueEndOp : XeMachine_Op<"payload_prologue_end",
    [Pure, Terminator, XeMachine_NoMachineInst, XeMachine_NoAsmEmission,
     HasParent<"PayloadPrologueOp">,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                               ["getMutableSuccessorOperands"]>]> {
  let summary = "mark the alternate kernel payload entry";
  let assemblyFormat = "attr-dict";
}

// Structured control flow. exec_if = mask-stack push/pop; uniform_if =
// scalar-condition branch. The emitter sees control flow only as these
// region ops.
class XeMachine_IfOp<string mnemonic> :
    XeMachine_Op<mnemonic, [SingleBlock,
                            DeclareOpInterfaceMethods<RegionBranchOpInterface,
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
    [SingleBlock, DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                             ["getEntrySuccessorOperands",
                                              "getSuccessorInputs"]>,
     DeclareOpInterfaceMethods<LoopLikeOpInterface,
                               ["getInitsMutable", "getLoopRegions",
                                "getLoopResults", "getRegionIterArgs",
                                "getYieldedValuesMutable"]>]> {
  let summary = "structured loop with explicit carried values";
  let arguments = (ins Variadic<AnyType>:$initialValues);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
  let assemblyFormat =
      "`(` $initialValues `)` $body attr-dict `:` functional-type($initialValues, $results)";
}

def XeMachine_YieldOp : XeMachine_Op<"yield",
    [Pure, Terminator, ReturnLike, XeMachine_NoMachineInst,
     XeMachine_NoAsmEmission,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                               ["getMutableSuccessorOperands"]>]> {
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

def XeMachine_MulOp :
    XeMachine_ALUOp<"mul", [], "accumulatorArithmetic", "mul">,
    XeMachine_SWSBFields {
  let summary = "multiply, accumulator destination";
  let arguments = !con((ins TypeAttr:$elemType,
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
                       XeMachine_RegOrImm:$src0, XeMachine_RegOrImm:$src1), swsbFields);
  let results = (outs XeMachine_ARFType:$dst);
  let assemblyFormat =
      "$src0 `,` $src1 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` $elemType `)` `->` qualified(type($dst))";
}

def XeMachine_DpasOp : XeMachine_Op<"dpas", [NoMemoryEffect,
    DeclareOpInterfaceMethods<XeMachine_RegisterStorageAliasOpInterface>,
    DeclareOpInterfaceMethods<XeMachine_SWSBInfoOpInterface>,
    DeclareOpInterfaceMethods<XeMachine_AsyncScoreboardOpInterface>],
    "systolic", "dpas">, XeMachine_SWSBFields {
  let summary = "SIMD16 DPAS matrix multiply-accumulate";
  let arguments = !con((ins XeMachine_RegType:$a, XeMachine_RegType:$b,
                       XeMachine_RegType:$acc,
                       XeMachine_DpasPrecision:$aPrecision,
                       XeMachine_DpasPrecision:$bPrecision,
                       DefaultValuedAttr<I32Attr, "8">:$systolicDepth,
                       DefaultValuedAttr<I32Attr, "8">:$repeatCount,
                       TypeAttr:$elemType), swsbFields);
  let results = (outs XeMachine_RegType:$dst);
  let assemblyFormat = "$a `,` $b `,` $acc attr-dict `:` `(` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($acc)) `)` `->` qualified(type($dst))";
  let hasVerifier = 1;
}

// Compare: writes a flag register, no GRF destination.
// cmp (exec|M0) (gt)f0.0 null<1>:ud src0 src1
def XeMachine_CmpOp : XeMachine_Op<"cmp", [NoMemoryEffect,
    DeclareOpInterfaceMethods<XeMachine_ALUOpInterface>,
    DeclareOpInterfaceMethods<XeMachine_SWSBInfoOpInterface>], "arfWrite",
    "cmp">, XeMachine_SWSBFields {
  let summary = "compare into a flag register";
  let arguments = !con((ins XeMachine_CondModifier:$cond,
                       TypeAttr:$elemType,
                       DefaultValuedAttr<I32Attr, "16">:$execSize,
                       OptionalAttr<XeMachine_RegionAttr>:$src0Region,
                       OptionalAttr<XeMachine_RegionAttr>:$src1Region,
                       OptionalAttr<I32Attr>:$src0Sub,
                       OptionalAttr<I32Attr>:$src1Sub,
                       OptionalAttr<TypeAttr>:$src0Type,
                       OptionalAttr<TypeAttr>:$src1Type,
                       XeMachine_RegOrImm:$src0, XeMachine_RegOrImm:$src1), swsbFields);
  let results = (outs XeMachine_ARFType:$flag);
  let assemblyFormat =
      "$cond $src0 `,` $src1 attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `,` $elemType `)` `->` qualified(type($flag))";
}

// Architectural register reference (r0 header, r1 inline args). The result
// carries a fixed physical base; regalloc must not move it.
def XeMachine_ArchRegOp : XeMachine_Op<"archreg", [NoMemoryEffect,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission]> {
  let summary = "reference an architectural register (r0/r1)";
  let arguments = (ins I32Attr:$index);
  let results = (outs XeMachine_RegType:$result);
  let assemblyFormat = "$index attr-dict `:` qualified(type($result))";
}

// ARF register reference (acc, flags, ...).
def XeMachine_ArfRegOp : XeMachine_Op<"arfreg", [NoMemoryEffect,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission]> {
  let summary = "reference an ARF register";
  let arguments = (ins XeMachine_ARFFile:$file, I32Attr:$index);
  let results = (outs XeMachine_ARFType:$result);
  let assemblyFormat = "$file `,` $index attr-dict `:` qualified(type($result))";
}

// Zero-cost views between contiguous register storage and mixed-width pieces.
// Alias offsets are cumulative dword widths, matching RegType::widthDwords.
def XeMachine_TupleToElementsOp : XeMachine_Op<"tuple_to_elements", [Pure,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission,
    DeclareOpInterfaceMethods<XeMachine_RegisterStorageAliasOpInterface>]> {
  let summary = "split register storage into mixed-width pieces";
  let arguments = (ins XeMachine_RegType:$tuple);
  let results = (outs Variadic<XeMachine_RegType>:$elements);
  let assemblyFormat = "$tuple attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def XeMachine_TupleFromElementsOp : XeMachine_Op<"tuple_from_elements", [Pure,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission,
    DeclareOpInterfaceMethods<XeMachine_RegisterStorageAliasOpInterface>]> {
  let summary = "concatenate mixed-width pieces into register storage";
  let arguments = (ins Variadic<XeMachine_RegType>:$elements);
  let results = (outs XeMachine_RegType:$tuple);
  let assemblyFormat = "$elements attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def XeMachine_UpdateTupleOp : XeMachine_Op<"update_tuple", [Pure,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission,
    DeclareOpInterfaceMethods<XeMachine_RegisterStorageAliasOpInterface>]> {
  let summary = "replace selected tuple slots while preserving storage";
  let arguments = (ins XeMachine_RegType:$base,
                        Variadic<XeMachine_RegType>:$updates,
                        ArrayAttr:$offsets);
  let results = (outs XeMachine_RegType:$result);
  let assemblyFormat =
      "$base `,` $updates attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// Token algebra. token: entry source. token_join: merge completions.
// after: issue-order sequencing without a completion promise.
def XeMachine_TokenOp : XeMachine_Op<"token", [NoMemoryEffect,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission]> {
  let summary = "initial memory-ordering token";
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "attr-dict";
}

def XeMachine_TokenJoinOp : XeMachine_Op<"token_join", [NoMemoryEffect,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission]> {
  let summary = "merge memory-ordering tokens";
  let arguments = (ins Variadic<XeMachine_MemTokenType>:$dependencies);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "$dependencies attr-dict `:` qualified(type($dependencies))";
}

def XeMachine_AfterOp : XeMachine_Op<"after", [NoMemoryEffect,
    XeMachine_NoMachineInst, XeMachine_NoAsmEmission,
    XeMachine_CompletionFree]> {
  let summary = "issue-order sequencing of memory ops";
  let arguments = (ins Variadic<XeMachine_MemTokenType>:$dependencies);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "$dependencies attr-dict `:` qualified(type($dependencies))";
}

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEOPS_TD
