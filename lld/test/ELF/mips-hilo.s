# REQUIRES: mips
# Check R_MIPS_HI16 / LO16 relocations calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { . = 0x20000; .text ALIGN(0x1000) : { *(.text) } }" > %t.script
# RUN: ld.lld %t.o --script %t.script -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex -d -t --no-show-raw-insn %t.exe | FileCheck %s

  .text
  .globl  __start
__start:
  lui    $t0,%hi(__start)
  lui    $t1,%hi(g1)
  addi   $t0,$t0,%lo(__start+4)
  addi   $t0,$t0,%lo(g1+8)

  lui    $t0,%hi(l1+0x10000)
  lui    $t1,%hi(l1+0x20000)
  addi   $t0,$t0,%lo(l1+(-4))

  .data
  .type  l1,@object
  .size  l1,4
l1:
  .word 0

  .globl g1
  .type  g1,@object
  .size  g1,4
g1:
  .word 0

# CHECK: SYMBOL TABLE:
# CHECK: 0021020 l     O .data   00000004 l1
# CHECK: 0021000 g       .text   00000000 __start
# CHECK: 0021024 g     O .data   00000004 g1

# CHECK:      <__start>:
# CHECK-NEXT:  21000:   lui    $8, 2
#                                  ^-- %hi(__start+4)
# CHECK-NEXT:  21004:   lui    $9, 2
#                                  ^-- %hi(g1+8)
# CHECK-NEXT:  21008:   addi   $8, $8, 4100
#                                      ^-- %lo(__start+4)
# CHECK-NEXT:  2100c:   addi   $8, $8, 4140
#                                      ^-- %lo(g1+8)
# CHECK-NEXT:  21010:   lui    $8, 3
#                                  ^-- %hi(l1+0x10000-4)
# CHECK-NEXT:  21014:   lui    $9, 4
#                                  ^-- %hi(l1+0x20000-4)
# CHECK-NEXT:  21018:   addi   $8, $8, 4124
#                                      ^-- %lo(l1-4)
