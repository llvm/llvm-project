# REQUIRES: mips
# Check handling of relocations against __gnu_local_gp symbol.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .text ALIGN(0x1000) : { *(.text) } \
# RUN:         . = 0x30000; .got :  { *(.got) } \
# RUN:       }" > %t.script
# RUN: ld.lld --script %t.script -o %t.exe %t.o
# RUN: llvm-objdump --no-print-imm-hex -d -t --no-show-raw-insn %t.exe | FileCheck %s

# CHECK: 00037ff0 l .got  00000000 .hidden _gp
# CHECK: 00011000 g .text 00000000 __start

# CHECK:      <__start>:
# CHECK-NEXT:    lui   $8, 3
# CHECK-NEXT:    addi  $8, $8, 32752

  .text
  .globl  __start
__start:
  lui    $t0,%hi(__gnu_local_gp)
  addi   $t0,$t0,%lo(__gnu_local_gp)
