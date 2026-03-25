# This test checks that reordering blocks which contain short range
# conditional branches may break if the target goes out of range.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --reorder-blocks=ext-tsp --compact-code-model
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=256
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --reorder-blocks=ext-tsp --compact-code-model --keep-nops
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=BEYOND-1KB

  .globl reorder_blocks
  .type reorder_blocks, %function
reorder_blocks:
.entry:
# FDATA: 1 reorder_blocks #.entry# 10
    cbgt x0, #0, .cold_exit
.fall_through:
# FDATA: 1 reorder_blocks #.fall_through# 10
    b .hot_exit
.cold_exit:
# FDATA: 1 reorder_blocks #.cold_exit# 1
    mov x0, #1
    ret
.hot_exit:
# FDATA: 1 reorder_blocks #.hot_exit# 10
  .rept NUM_NOPS
    nop
  .endr
    mov x0, #2
    ret

## Force relocation mode.
.reloc 0, R_AARCH64_NONE

# CHECK: Disassembly of section .text:

# CHECK:      <reorder_blocks>:
# CHECK-NEXT:           {{.*}} cbgt x0, #0x0, 0x[[ADDR:[0-9a-f]+]] <{{.*}}>
# CHECK:      <.hot_exit>:
# CHECK-NEXT:           {{.*}} mov  x0, #0x2 // =2
# CHECK-NEXT:           {{.*}} ret
# CHECK:      <.cold_exit>:
# CHECK-NEXT: [[ADDR]]: {{.*}} mov  x0, #0x1 // =1
# CHECK-NEXT:           {{.*}} ret

# BEYOND-1KB: Disassembly of section .text:

# BEYOND-1KB:      <reorder_blocks>:
# BEYOND-1KB-NEXT:            {{.*}} cblt x0, #0x1, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# BEYOND-1KB-NEXT:            {{.*}} b              0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# BEYOND-1KB:      <.hot_exit>:
# BEYOND-1KB-NEXT: [[ADDR0]]: {{.*}} nop
# BEYOND-1KB:                 {{.*}} mov  x0, #0x2 // =2
# BEYOND-1KB-NEXT:            {{.*}} ret
# BEYOND-1KB:      <.cold_exit>:
# BEYOND-1KB-NEXT: [[ADDR1]]: {{.*}} mov  x0, #0x1 // =1
# BEYOND-1KB-NEXT:            {{.*}} ret
