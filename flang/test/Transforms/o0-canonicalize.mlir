// RUN: fir-opt %s --fir-o0-canonicalize | FileCheck %s --check-prefix=O0 --implicit-check-not=arith.addi --implicit-check-not=arith.muli --implicit-check-not=func.call_indirect
// RUN: fir-opt %s --canonicalize | FileCheck %s --check-prefix=CANONICAL

// Other dialects' canonicalization patterns, folding, and dead operation
// removal still run. In particular, the indirect-to-direct call rewrite is
// implemented by CallIndirectOp::canonicalize, which TableGen registers as a
// rewrite pattern, not a fold. CF methods and explicit patterns, including
// constant branch simplification and single-predecessor merging, are omitted.
// O0-LABEL: func.func @cleanup(
// O0-SAME: %[[X:.*]]: i32)
// O0: %[[TRUE:.*]] = arith.constant true
// O0: %[[VALUE:.*]] = call @callee(%[[X]]) : (i32) -> i32
// O0-NEXT: cf.cond_br %[[TRUE]], ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// O0: ^[[THEN]]:
// O0-NEXT: cf.br ^[[JOIN:bb[0-9]+]]
// O0: ^[[ELSE]]:
// O0-NEXT: cf.br ^[[JOIN]]
// O0: ^[[JOIN]]:
// O0-NEXT: return %[[VALUE]] : i32
// CANONICAL-LABEL: func.func @cleanup(
// CANONICAL-SAME: %[[X:.*]]: i32)
// CANONICAL-NEXT: %[[VALUE:.*]] = call @callee(%[[X]]) : (i32) -> i32
// CANONICAL-NEXT: return %[[VALUE]] : i32
func.func @cleanup(%x: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  %sum = arith.addi %x, %zero : i32
  %dead = arith.muli %x, %x : i32
  %callee = func.constant @callee : (i32) -> i32
  %value = func.call_indirect %callee(%sum) : (i32) -> i32
  cf.cond_br %true, ^then, ^else
^then:
  cf.br ^join
^else:
  cf.br ^join
^join:
  return %value : i32
}
func.func private @callee(i32) -> i32
