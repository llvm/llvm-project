# REQUIRES: mips
# Check setup of GP relative offsets in a function's prologue.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { foo = 0x2000; _gp = 0x3000; }" > %t.script
# RUN: ld.lld %t.o --script %t.script -shared -o %t.so
# RUN: llvm-objdump -d -t --print-imm-hex --no-show-raw-insn %t.so | FileCheck %s

# CHECK:      {{.*}}  lui     $gp, 0x0
# CHECK-NEXT: {{.*}}  daddu   $gp, $gp, $25
# CHECK-NEXT: {{.*}}  daddiu  $gp, $gp, 0x1000

  .text
  lui     $gp,%hi(%neg(%gp_rel(foo)))
  daddu   $gp,$gp,$t9
  daddiu  $gp,$gp,%lo(%neg(%gp_rel(foo)))
