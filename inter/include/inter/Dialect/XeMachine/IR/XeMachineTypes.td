// xemachine types. Virtual and physical registers share a type; regalloc
// rewrites the index slot (see inter/docs/BackendDialect.md).

#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINETYPES_TD
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINETYPES_TD

include "inter/Dialect/XeMachine/IR/XeMachineBase.td"

// GRF storage. widthDwords = size in 32-bit units (Xe2 GRF = 16 dwords).
// baseGRF = physical base register, -1 = virtual.
def XeMachine_RegType : TypeDef<XeMachineDialect, "Reg"> {
  let mnemonic = "reg";
  let summary = "GRF storage: width in dwords, physical base GRF or virtual";
  let parameters = (ins "uint32_t":$widthDwords, "int32_t":$baseGRF);
  let assemblyFormat = "`<` $widthDwords `,` $baseGRF `>`";
}

// ARF storage: address registers, flags, accumulators, specials.
def XeMachine_ARFType : TypeDef<XeMachineDialect, "ARF"> {
  let mnemonic = "arf";
  let summary = "ARF storage: file selector, width in dwords, index or virtual";
  let parameters = (ins "ARFFile":$file, "uint32_t":$widthDwords,
                      "int32_t":$index);
  let assemblyFormat = "`<` $file `,` $widthDwords `,` $index `>`";
}

// SSA immediate. EU allows at most one immediate source per instruction;
// the constraint is checked by an op interface, not by the type.
def XeMachine_ImmType : TypeDef<XeMachineDialect, "Imm"> {
  let mnemonic = "imm";
  let summary = "SSA immediate; element type rides on the producing op";
  let assemblyFormat = "``";
}

// Explicit memory ordering token. Legality is SSA dominance plus token
// edges; see inter/docs/Synchronization.md.
def XeMachine_MemTokenType : TypeDef<XeMachineDialect, "MemToken"> {
  let mnemonic = "mem.token";
  let summary = "Memory ordering token";
  let assemblyFormat = "``";
}

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINETYPES_TD
