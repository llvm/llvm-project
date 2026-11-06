## Inlining an lld Cortex-A53 erratum 843419 veneer has to reconnect the block
## to the veneer's branch-back target. The branch being replaced is a tail
## call, so the block has no successors in the CFG; without an explicit edge,
## block reordering is free to move the two apart and nothing adds a branch to
## get back, because a block with no successors has no fall-through to fix up.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --entry=_start --fix-cortex-a53-843419

## The input must carry an lld veneer, otherwise the test proves nothing.
# RUN: llvm-nm %t.exe | FileCheck %s --check-prefix=CHECK-INPUT
# CHECK-INPUT: __CortexA53843419_

## The block holding the inlined load continues at the branch-back target.
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 \
# RUN:   --drop-cortex-a53-843419-veneers --print-veneer-elimination \
# RUN:   --print-only=_start | FileCheck %s --check-prefix=CHECK-CFG
# CHECK-CFG:      ldr x3, [x0
# CHECK-CFG-NEXT: Successors: [[RET:[.A-Za-z0-9_]+]]
# CHECK-CFG:      [[RET]] ({{.*}} instructions
# CHECK-CFG:      Predecessors: {{.*}}, {{.*}}

## And once reordering separates them, the branch is materialized rather than
## the block running into whatever was laid out next.
# RUN: llvm-bolt %t.exe -o %t.reverse.bolt --lite=0 \
# RUN:   --drop-cortex-a53-843419-veneers --reorder-blocks=reverse
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.reverse.bolt \
# RUN:   | FileCheck %s
# CHECK-LABEL: <_start>:
# CHECK-NOT:   b {{.*}} <__CortexA53843419_
# CHECK:       ldr x3, [x0
# CHECK-NEXT:  b {{.*}} <_start+

  .text
  .globl _start
  .type _start, %function
  .p2align 12
_start:
  cbz x5, .Lpatched
  mov x3, #7
  b .Lafter
.Lpatched:
## Push the ADRP to page offset 0xff8 so lld patches the final load.
  .rept 1019
  nop
  .endr
  adrp x0, data
  ldr x1, [x2]
  ldr x3, [x0, :lo12:data]
.Lafter:
  add x0, x3, #1
  ret
  nop
  .size _start, .-_start

  .data
  .p2align 3
data:
  .xword 42

  .text
## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
