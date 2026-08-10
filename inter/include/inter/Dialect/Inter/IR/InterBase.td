// The inter frontend dialect: kernel-level ops with explicit memory
// ordering tokens (design doc section 9). Lives between the llvm dialect
// (plus scf) and xemachine selection. Tokens reuse !xemachine.mem.token;
// the AA-driven conversion pass decides which edges exist.

#ifndef INTER_DIALECT_INTER_IR_INTERBASE_TD
#define INTER_DIALECT_INTER_IR_INTERBASE_TD

include "mlir/IR/OpBase.td"

def XWDialect : Dialect {
  let name = "xw";
  let summary = "Kernel-level frontend ops with explicit memory tokens";
  let description = [{
    Load/store/barrier/atomic with token operands and results, work-item id
    special ops, and the token algebra. Legality is SSA dominance plus
    explicit token edges; no pass below the AA conversion may infer ordering.
  }];
  let cppNamespace = "::xw";
}

#endif // INTER_DIALECT_INTER_IR_INTERBASE_TD
