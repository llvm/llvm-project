// inter frontend ops. Memory ops carry an optional dependency token and
// return a result token; ordering edges are produced by the
// inter-convert-memory pass from alias analysis, never inferred later.

#ifndef INTER_DIALECT_INTER_IR_INTEROPS_TD
#define INTER_DIALECT_INTER_IR_INTEROPS_TD

include "inter/Dialect/Inter/IR/InterBase.td"
include "inter/Dialect/XeMachine/IR/XeMachineTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class XW_Op<string mnemonic, list<Trait> traits = []> :
    Op<XWDialect, mnemonic, traits> {
  let cppNamespace = "::xw";
}

// Memory ordering token: the entry source for a kernel's token graph.
def XW_TokenOp : XW_Op<"token", [NoMemoryEffect]> {
  let summary = "initial memory-ordering token";
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "attr-dict";
}

def XW_TokenJoinOp : XW_Op<"token_join", [NoMemoryEffect]> {
  let summary = "merge memory-ordering tokens";
  let arguments = (ins Variadic<XeMachine_MemTokenType>:$dependencies);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "$dependencies attr-dict `:` type($dependencies)";
}

// Global memory load (A64 stateless at machine level).
def XW_LoadOp : XW_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "memory load with an ordering token";
  let arguments = (ins AnyType:$address,
                       Optional<XeMachine_MemTokenType>:$dependency);
  let results = (outs AnyType:$value, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$address (`dep` $dependency^)? attr-dict `:` type($address) `->` type($value)";
}

def XW_StoreOp : XW_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "memory store with an ordering token";
  let arguments = (ins AnyType:$address, AnyType:$value,
                       Optional<XeMachine_MemTokenType>:$dependency);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$address `,` $value (`dep` $dependency^)? attr-dict `:` type($address) `,` type($value) `->` type($token)";
}

// atomic_add returning the old value.
def XW_AtomicAddOp : XW_Op<"atomic_add",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "atomic add with an ordering token";
  let arguments = (ins AnyType:$address, AnyType:$value,
                       Optional<XeMachine_MemTokenType>:$dependency);
  let results = (outs AnyType:$old, XeMachine_MemTokenType:$token);
  let assemblyFormat =
      "$address `,` $value (`dep` $dependency^)? attr-dict `:` type($address) `,` type($value) `->` `(` type($old) `,` type($token) `)`";
}

// Workgroup barrier: orders against ALL live token chains (the conversion
// pass joins them into its dependency).
def XW_BarrierOp : XW_Op<"barrier",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "workgroup barrier with an ordering token";
  let arguments = (ins Optional<XeMachine_MemTokenType>:$dependency);
  let results = (outs XeMachine_MemTokenType:$token);
  let assemblyFormat = "(`dep` $dependency^)? attr-dict `:` type($token)";
}

// Work-item id special ops. The id is a per-lane value.
def XW_GlobalIdOp : XW_Op<"global_id", [NoMemoryEffect]> {
  let summary = "global work-item id";
  let arguments = (ins I32Attr:$dim);
  let results = (outs AnyType:$id);
  let assemblyFormat = "$dim attr-dict `:` type($id)";
}

def XW_LocalIdOp : XW_Op<"local_id", [NoMemoryEffect]> {
  let summary = "local work-item id";
  let arguments = (ins I32Attr:$dim);
  let results = (outs AnyType:$id);
  let assemblyFormat = "$dim attr-dict `:` type($id)";
}

#endif // INTER_DIALECT_INTER_IR_INTEROPS_TD
