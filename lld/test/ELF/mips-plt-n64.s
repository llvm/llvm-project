# REQUIRES: mips
# Check R_MIPS_26 relocation handling in case of N64 ABIs.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .text ALIGN(0x10000) : { *(.text) } \
# RUN:         . = 0x30000; .data                : { *(.data) } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         %S/Inputs/mips-dynamic.s -o %t-so.o
# RUN: ld.lld %t-so.o -shared -soname=t.so -o %t.so
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o %t.so -script %t.script -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.exe \
# RUN:   | FileCheck %s --check-prefixes=CHECK,DEFAULT
# RUN: ld.lld %t-so.o -shared -soname=t.so -o %t.so -z hazardplt
# RUN: ld.lld %t.o %t.so -script %t.script -o %t.exe -z hazardplt
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.exe \
# RUN:   | FileCheck %s --check-prefixes=CHECK,HAZARDPLT

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <__start>:
# CHECK-NEXT:    20000:       jal     0x20030
# CHECK-NEXT:    20004:       nop
# CHECK-EMPTY:
# CHECK-NEXT: Disassembly of section .plt:
# CHECK-EMPTY:
# CHECK-NEXT: <.plt>:
# CHECK-NEXT:    20010:       lui     $14, 3
# CHECK-NEXT:    20014:       ld      $25, 8($14)
# CHECK-NEXT:    20018:       addiu   $14, $14, 8
# CHECK-NEXT:    2001c:       subu    $24, $24, $14
# CHECK-NEXT:    20020:       move    $15, $ra
# CHECK-NEXT:    20024:       srl     $24, $24, 3
# DEFAULT:       20028:       jalr    $25
# HAZARDPLT:     20028:       jalr.hb $25
# CHECK-NEXT:    2002c:       addiu   $24, $24, -2
# CHECK-NEXT:    20030:       lui     $15, 3
# CHECK-NEXT:    20034:       ld      $25, 24($15)
# DEFAULT:       20038:       jr      $25
# HAZARDPLT:     20038:       jr.hb   $25
# CHECK-NEXT:    2003c:       daddiu  $24, $15, 24

  .text
  .option pic0
  .global __start
__start:
  jal   foo0
