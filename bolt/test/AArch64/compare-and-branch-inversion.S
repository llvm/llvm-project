# This test checks that branch inversion works when reordering blocks which
# contain short range conditional branches. Handles edge cases, like when
# the immediate value is the upper or lower allowed value in which case the
# transformation bails. If liveness analysis proves that the condition flags
# are dead we can replace the branch with cmp + b.cc

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
#
# RUN: llvm-bolt -v=1 %t -o %t.bolt --data %t.fdata --reorder-blocks=ext-tsp --compact-code-model \
# RUN:   | FileCheck %s --check-prefix=BOLT-INFO-NO-LIVENESS
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=COMMON --check-prefix=NO-LIVENESS
#
# RUN: llvm-bolt -v=1 %t -o %t.bolt --data %t.fdata --reorder-blocks=ext-tsp --compact-code-model \
# RUN:   --liveness-analysis | FileCheck %s --check-prefix=BOLT-INFO-LIVENESS
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=COMMON --check-prefix=LIVENESS

# COMMON: Disassembly of section .text:

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

# COMMON: <immediate_increment>:
# COMMON-NEXT:            {{.*}} cblt x0, #0x1, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# COMMON-NEXT:            {{.*}} mov  x0, #0x2 // =2
# COMMON-NEXT:            {{.*}} ret
# COMMON-NEXT: [[ADDR0]]: {{.*}} mov x0, #0x1 // =1
# COMMON-NEXT:            {{.*}} ret

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

# COMMON: <immediate_decrement>:
# COMMON-NEXT:            {{.*}} cbhi x0, #0x0, 0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# COMMON-NEXT:            {{.*}} mov  x0, #0x2 // =2
# COMMON-NEXT:            {{.*}} ret
# COMMON-NEXT: [[ADDR1]]: {{.*}} mov x0, #0x1 // =1
# COMMON-NEXT:            {{.*}} ret

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

# COMMON: <register_swap>:
# COMMON-NEXT:            {{.*}} cbgt x1, x0, 0x[[ADDR2:[0-9a-f]+]] <{{.*}}>
# COMMON-NEXT:            {{.*}} mov  x0, #0x2 // =2
# COMMON-NEXT:            {{.*}} ret
# COMMON-NEXT: [[ADDR2]]: {{.*}} mov x0, #0x1 // =1
# COMMON-NEXT:            {{.*}} ret

  .globl immediate_overflow
  .type immediate_overflow, %function
immediate_overflow:
.entry3:
# FDATA: 1 immediate_overflow #.entry3# 10
    cbgt x0, #63, .exit3
.cold3:
# FDATA: 1 immediate_overflow #.cold3# 1
    mov x0, #1
    ret
.exit3:
# FDATA: 1 immediate_overflow #.exit3# 10
    mov x0, #2
    ret

# BOLT-INFO-NO-LIVENESS: unable to swap successors in immediate_overflow
#
# Without liveness the blocks get reordered, but since the branch is
# irreversible an additional unconditional branch is emitted.
# This codegen is suboptimal yet correct.
#
# NO-LIVENESS: <immediate_overflow>:
# NO-LIVENESS-NEXT:            {{.*}} cbgt x0, #0x3f, 0x[[ADDR3:[0-9a-f]+]] <{{.*}}>
# NO-LIVENESS-NEXT:            {{.*}} b               0x[[ADDR4:[0-9a-f]+]] <{{.*}}>
# NO-LIVENESS-NEXT: [[ADDR3]]: {{.*}} mov  x0, #0x2 // =2
# NO-LIVENESS-NEXT:            {{.*}} ret
# NO-LIVENESS-NEXT: [[ADDR4]]: {{.*}} mov x0, #0x1 // =1
# NO-LIVENESS-NEXT:            {{.*}} ret

# LIVENESS: <immediate_overflow>:
# LIVENESS-NEXT:            {{.*}} cmp  x0, #0x3f
# LIVENESS-NEXT:            {{.*}} b.le 0x[[ADDR5:[0-9a-f]+]] <{{.*}}>
# LIVENESS-NEXT:            {{.*}} mov  x0, #0x2 // =2
# LIVENESS-NEXT:            {{.*}} ret
# LIVENESS-NEXT: [[ADDR5]]: {{.*}} mov x0, #0x1 // =1
# LIVENESS-NEXT:            {{.*}} ret

  .globl irreversible
  .type irreversible, %function
irreversible:
.entry4:
# FDATA: 1 irreversible #.entry4# 10
    cmp x0, #63
    cbgt x0, #63, .exit4
.cold4:
# FDATA: 1 irreversible #.cold4# 1
    csel x0, x1, x2, le
    ret
.exit4:
# FDATA: 1 irreversible #.exit4# 10
    mov x0, #2
    ret

# BOLT-INFO-NO-LIVENESS: unable to swap successors in irreversible
# BOLT-INFO-LIVENESS:    unable to swap successors in irreversible

# COMMON: <irreversible>:
# COMMON-NEXT:            {{.*}} cmp  x0, #0x3f
# COMMON-NEXT:            {{.*}} cbgt x0, #0x3f, 0x[[ADDR6:[0-9a-f]+]] <{{.*}}>
# COMMON-NEXT:            {{.*}} b               0x[[ADDR7:[0-9a-f]+]] <{{.*}}>
# COMMON-NEXT: [[ADDR6]]: {{.*}} mov  x0, #0x2 // =2
# COMMON-NEXT:            {{.*}} ret
# COMMON-NEXT: [[ADDR7]]: {{.*}} csel x0, x1, x2, le
# COMMON-NEXT:            {{.*}} ret

## Force relocation mode.
.reloc 0, R_AARCH64_NONE
