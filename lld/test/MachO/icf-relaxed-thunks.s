# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -dylib --icf=relaxed_thunks -interposable %t.o -map - -o %t.dylib | FileCheck %s
# RUN: llvm-objdump --macho --disassemble %t.dylib | FileCheck %s --check-prefix=DISASM

## Taking the leaves' addresses must still produce different code. A caller of
## a thunk leading to a different body must also keep its own body.
# CHECK-DAG: 0x0000000C {{.*}} _address_a
# CHECK-DAG: 0x0000000C {{.*}} _address_b
# CHECK-DAG: 0x00000008 {{.*}} _leaf_c
# CHECK-DAG: 0x00000004 {{.*}} _leaf_d
# CHECK-DAG: 0x00000004 {{.*}} _interposable_leaf
# CHECK-DAG: 0x00000018 {{.*}} _call_c

## Calls and tail branches to equivalent bodies allow their callers to fold.
# CHECK-DAG: [[CALL:0x[0-9A-Fa-f]+]] 0x00000018 {{.*}} _call_a
# CHECK-DAG: [[CALL]] 0x00000000 {{.*}} _call_b
# CHECK-DAG: [[TAIL:0x[0-9A-Fa-f]+]] 0x00000008 {{.*}} _tail_a
# CHECK-DAG: [[TAIL]] 0x00000000 {{.*}} _tail_b

## An equivalent caller of an interposable leaf must keep its own body.
# CHECK-DAG: 0x00000018 {{.*}} _call_interposable

## Address-significant callers must retain unique entrypoints too.
# CHECK-DAG: 0x00000018 {{.*}} _unique_a
# CHECK-DAG: 0x00000004 {{.*}} _unique_b

## Recursive bodies can be equivalent while their entrypoints remain distinct.
# CHECK-DAG: 0x00000010 {{.*}} _recursive_a
# CHECK-DAG: 0x00000004 {{.*}} _recursive_b

## Alternate entries prevent the targets from folding. Both callers reference
## the same targets in opposite order, so their initial hashes are equal.
## The first variable pass must distinguish their targets using icfEqClass[1].
# CHECK-DAG: 0x00000008 {{.*}} _non_foldable_a
# CHECK-DAG: 0x00000008 {{.*}} _non_foldable_b
# CHECK-DAG: 0x00000014 {{.*}} _call_non_foldable_ab
# CHECK-DAG: 0x00000014 {{.*}} _call_non_foldable_ba

## Address-significant leaves share a body through a unique entry thunk.
# DISASM-LABEL: _leaf_a:
# DISASM-NEXT: [[#%x,LEAF_A:]]:

## Folded callers share a callsite; distinct callees keep distinct callers.
# DISASM-LABEL: _call_b:
# DISASM: {{.*}} bl _leaf_a
# DISASM-LABEL: _call_c:
# DISASM: {{.*}} bl _leaf_d

# DISASM-LABEL: _call_only_b:
# DISASM: {{.*}} bl _leaf_b

# DISASM-LABEL: _address_a:
# DISASM: add x0, x0, #0x[[#%x,LEAF_A]]
# DISASM-LABEL: _address_b:
# DISASM: add x0, x0, #0x[[#%x,LEAF_B:]]
## Direct and indirect calls to a thunk preserve the same entrypoint.
# DISASM-LABEL: _call_direct_and_indirect_b:
# DISASM: add x8, x8, #0x[[#%x,LEAF_B]]
# DISASM: {{.*}} bl _leaf_b
# DISASM: blr x8

## Runtime-bound direct calls still resolve the original symbol.
# DISASM-LABEL: _call_interposable:
# DISASM: {{.*}} bl {{.*}}symbol stub for: _interposable_leaf

# DISASM-LABEL: _leaf_d:
# DISASM-NEXT: {{.*}} b _leaf_c
# DISASM-LABEL: _leaf_b:
# DISASM-NEXT: [[#%x,LEAF_B]]: {{.*}} b _leaf_a
# DISASM-LABEL: _unique_b:
# DISASM-NEXT: {{.*}} b _unique_a

.text
.globl _interposable_leaf

_leaf_a:
  add w0, w0, #7
  ret

_leaf_b:
  add w0, w0, #7
  ret

_leaf_c:
  add w0, w0, #9
  ret

_leaf_d:
  add w0, w0, #9
  ret

_call_a:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl _leaf_a
  add w0, w0, #3
  ldp x29, x30, [sp], #16
  ret

_call_b:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl _leaf_b
  add w0, w0, #3
  ldp x29, x30, [sp], #16
  ret

_call_c:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl _leaf_d
  add w0, w0, #3
  ldp x29, x30, [sp], #16
  ret

_call_only_b:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl _leaf_b
  add w0, w0, #11
  ldp x29, x30, [sp], #16
  ret

_tail_a:
  eor w0, w0, #1
  b _leaf_a

_tail_b:
  eor w0, w0, #1
  b _leaf_b

_unique_a:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl _leaf_a
  add w0, w0, #5
  ldp x29, x30, [sp], #16
  ret

_unique_b:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl _leaf_b
  add w0, w0, #5
  ldp x29, x30, [sp], #16
  ret

_recursive_a:
  subs w0, w0, #1
  b.eq Lrecursive_done_a
  b _recursive_step_a
Lrecursive_done_a:
  ret

_recursive_b:
  subs w0, w0, #1
  b.eq Lrecursive_done_b
  b _recursive_step_b
Lrecursive_done_b:
  ret

_recursive_step_a:
  b _recursive_a
  nop

_recursive_step_b:
  b _recursive_b
  nop

_address_a:
  adrp x0, _leaf_a@PAGE
  add x0, x0, _leaf_a@PAGEOFF
  ret

_address_b:
  adrp x0, _leaf_b@PAGE
  add x0, x0, _leaf_b@PAGEOFF
  ret

_call_direct_and_indirect_b:
  stp x29, x30, [sp, #-32]!
  mov x29, sp
  adrp x8, _leaf_b@PAGE
  add x8, x8, _leaf_b@PAGEOFF
  str x8, [sp, #16]
  bl _leaf_b
  ldr x8, [sp, #16]
  blr x8
  ldp x29, x30, [sp], #32
  ret

_interposable_leaf:
  add w0, w0, #7
  ret

_call_interposable:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl _interposable_leaf
  add w0, w0, #3
  ldp x29, x30, [sp], #16
  ret

.alt_entry _non_foldable_alt_a
_non_foldable_a:
  add w0, w0, #1
  nop
_non_foldable_alt_a:
  ret

.alt_entry _non_foldable_alt_b
_non_foldable_b:
  lsl w0, w0, #1
  nop
_non_foldable_alt_b:
  ret

_call_non_foldable_ab:
  stp x29, x30, [sp, #-16]!
  bl _non_foldable_a
  bl _non_foldable_b
  ldp x29, x30, [sp], #16
  ret

_call_non_foldable_ba:
  stp x29, x30, [sp, #-16]!
  bl _non_foldable_b
  bl _non_foldable_a
  ldp x29, x30, [sp], #16
  ret

.data
_addresses:
  .quad _leaf_a, _leaf_b, _leaf_c, _leaf_d, _unique_a, _unique_b, _recursive_a, _recursive_b

.addrsig
.addrsig_sym _leaf_a
.addrsig_sym _leaf_b
.addrsig_sym _leaf_c
.addrsig_sym _leaf_d
.addrsig_sym _unique_a
.addrsig_sym _unique_b
.addrsig_sym _recursive_a
.addrsig_sym _recursive_b
.addrsig_sym _recursive_step_a
.addrsig_sym _recursive_step_b
.addrsig_sym _interposable_leaf
.addrsig_sym _non_foldable_a
.addrsig_sym _non_foldable_b

.subsections_via_symbols
