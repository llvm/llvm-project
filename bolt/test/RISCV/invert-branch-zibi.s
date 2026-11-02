// This test checks that profile-guided (ext-tsp) basic block reordering
// correctly inverts the Zibi branch-with-immediate instructions
// (beqi/bnei) when the hot successor becomes the fallthrough block.

// RUN: %clang %cflags64 -march=rv64gc_zibi0p1 -menable-experimental-extensions \
// RUN:   -Wl,-q %s -o %t
// RUN: link_fdata --no-lbr %s %t %t.fdata
// RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --reorder-blocks=ext-tsp
// RUN: llvm-objdump -d --no-show-raw-insn --mattr=+experimental-zibi %t.bolt \
// RUN:   | FileCheck %s

  .globl invert_beqi
  .type invert_beqi, %function
invert_beqi:
.entry:
# FDATA: 1 invert_beqi #.entry# 10
  beqi t0, 1, .hot_exit
.fall_through:
# FDATA: 1 invert_beqi #.fall_through# 1
  li a0, 1
  ret
.hot_exit:
# FDATA: 1 invert_beqi #.hot_exit# 10
  li a0, 2
  ret
  .size invert_beqi, .-invert_beqi

  .globl invert_bnei
  .type invert_bnei, %function
invert_bnei:
.entry2:
# FDATA: 1 invert_bnei #.entry2# 10
  bnei t0, 1, .hot_exit2
.fall_through2:
# FDATA: 1 invert_bnei #.fall_through2# 1
  li a0, 1
  ret
.hot_exit2:
# FDATA: 1 invert_bnei #.hot_exit2# 10
  li a0, 2
  ret
  .size invert_bnei, .-invert_bnei

## Force relocation mode.
.reloc 0, R_RISCV_NONE

# CHECK: Disassembly of section .text:

# CHECK:      <invert_beqi>:
# CHECK-NEXT:            {{.*}} bnei t0, 0x1, 0x[[ADDR:[0-9a-f]+]] <{{.*}}>
# CHECK:                 {{.*}}li a0, 0x2
# CHECK-NEXT:            {{.*}} ret
# CHECK: [[ADDR]]:{{.*}}li a0, 0x1
# CHECK-NEXT:            {{.*}} ret

# CHECK:      <invert_bnei>:
# CHECK-NEXT:            {{.*}} beqi t0, 0x1, 0x[[ADDR2:[0-9a-f]+]] <{{.*}}>
# CHECK:                 {{.*}}li a0, 0x2
# CHECK-NEXT:            {{.*}} ret
# CHECK: [[ADDR2]]:{{.*}}li a0, 0x1
# CHECK-NEXT:            {{.*}} ret
