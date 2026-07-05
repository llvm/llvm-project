# REQUIRES: mips
# Check PLT entries generation in case of R6 ABI version.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .text ALIGN(0x10000) : { *(.text) } \
# RUN:         . = 0x30000; .data                : { *(.data) } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r6 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r6 %S/Inputs/mips-dynamic.s -o %t2.o
# RUN: ld.lld %t2.o -shared -soname=t.so -o %t.so
# RUN: ld.lld %t1.o %t.so -script %t.script -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.exe | FileCheck %s --check-prefixes=DEFAULT,CHECK
# RUN: ld.lld %t2.o -shared -soname=t.so -o %t.so -z hazardplt
# RUN: ld.lld %t1.o %t.so -script %t.script -o %t.exe -z hazardplt
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.exe | FileCheck %s --check-prefixes=HAZARDPLT,CHECK

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <__start>:
# CHECK-NEXT:   20000:       jal     0x20030
#                                    ^-- 0x20030 gotplt[foo0]
# CHECK-NEXT:   20004:       nop
#
# CHECK-EMPTY:
# CHECK-NEXT: Disassembly of section .plt:
# CHECK-EMPTY:
# CHECK-NEXT: <.plt>:
# CHECK-NEXT:   20010:       aui     $gp, $zero, 3
# CHECK-NEXT:   20014:       lw      $25, 4($gp)
# CHECK-NEXT:   20018:       addiu   $gp, $gp, 4
# CHECK-NEXT:   2001c:       subu    $24, $24, $gp
# CHECK-NEXT:   20020:       move    $15, $ra
# CHECK-NEXT:   20024:       srl     $24, $24, 2
# DEFAULT:      20028:       jalr    $25
# HAZARDPLT:    20028:       jalr.hb $25
# CHECK-NEXT:   2002c:       addiu   $24, $24, -2

# CHECK-NEXT:   20030:       aui     $15, $zero, 3
# CHECK-NEXT:   20034:       lw      $25, 12($15)
# DEFAULT:      20038:       jr      $25
# HAZARDPLT:    20038:       jr.hb   $25
# CHECK-NEXT:   2003c:       addiu   $24, $15, 12

  .text
  .global __start
__start:
  jal foo0        # R_MIPS_26 against 'foo0' from DSO
