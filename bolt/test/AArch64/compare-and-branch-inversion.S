# This test checks that branch inversion works when reordering blocks which
# contain short range conditional branches. Handles edge cases, like when
# the immediate value is the upper or lower allowed value in which case the
# transformation bails.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt -v=1 %t -o %t.bolt --data %t.fdata --reorder-blocks=ext-tsp --compact-code-model \
# RUN:   | FileCheck %s --check-prefix=BOLT-INFO
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# CHECK: Disassembly of section .text:

  .globl immediate_increment
  .type immediate_increment, %function
immediate_increment:
.entry0:
# FDATA: 1 immediate_increment #.entry0# 10
    cbgt x0, #0, .exit0
.cold0:
# FDATA: 1 immediate_increment #.cold0# 1
    mov x0, #1
    ret
.exit0:
# FDATA: 1 immediate_increment #.exit0# 10
    mov x0, #2
    ret

# CHECK: <immediate_increment>:
# CHECK-NEXT:            {{.*}} cblt x0, #0x1, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} mov  x0, #0x2 // =2
# CHECK-NEXT:            {{.*}} ret
# CHECK-NEXT: [[ADDR0]]: {{.*}} mov x0, #0x1 // =1
# CHECK-NEXT:            {{.*}} ret

  .globl immediate_decrement
  .type immediate_decrement, %function
immediate_decrement:
.entry1:
# FDATA: 1 immediate_decrement #.entry1# 10
    cblo x0, #1, .exit1
.cold1:
# FDATA: 1 immediate_decrement #.cold1# 1
    mov x0, #1
    ret
.exit1:
# FDATA: 1 immediate_decrement #.exit1# 10
    mov x0, #2
    ret

# CHECK: <immediate_decrement>:
# CHECK-NEXT:            {{.*}} cbhi x0, #0x0, 0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} mov  x0, #0x2 // =2
# CHECK-NEXT:            {{.*}} ret
# CHECK-NEXT: [[ADDR1]]: {{.*}} mov x0, #0x1 // =1
# CHECK-NEXT:            {{.*}} ret

  .globl register_swap
  .type register_swap, %function
register_swap:
.entry2:
# FDATA: 1 register_swap #.entry2# 10
    cbge x0, x1, .exit2
.cold2:
# FDATA: 1 register_swap #.cold2# 1
    mov x0, #1
    ret
.exit2:
# FDATA: 1 register_swap #.exit2# 10
    mov x0, #2
    ret

# CHECK: <register_swap>:
# CHECK-NEXT:            {{.*}} cbgt x1, x0, 0x[[ADDR2:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} mov  x0, #0x2 // =2
# CHECK-NEXT:            {{.*}} ret
# CHECK-NEXT: [[ADDR2]]: {{.*}} mov x0, #0x1 // =1
# CHECK-NEXT:            {{.*}} ret

  .globl irreversible
  .type irreversible, %function
irreversible:
.entry3:
# FDATA: 1 irreversible #.entry3# 10
    cbgt x0, #63, .exit3
.cold3:
# FDATA: 1 irreversible #.cold3# 1
    mov x0, #1
    ret
.exit3:
# FDATA: 1 irreversible #.exit3# 10
    mov x0, #2
    ret

# BOLT-INFO: unable to swap successors in irreversible

# CHECK: <irreversible>:
# CHECK-NEXT:            {{.*}} cbgt x0, #0x3f, 0x[[ADDR3:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} b               0x[[ADDR4:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT: [[ADDR3]]: {{.*}} mov  x0, #0x2 // =2
# CHECK-NEXT:            {{.*}} ret
# CHECK-NEXT: [[ADDR4]]: {{.*}} mov x0, #0x1 // =1
# CHECK-NEXT:            {{.*}} ret

## Force relocation mode.
.reloc 0, R_AARCH64_NONE
