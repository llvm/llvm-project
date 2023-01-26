# REQUIRES: mips
# Check R_MIPS_PCxxx relocations calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r6 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r6 %S/Inputs/mips-dynamic.s -o %t2.o
# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .text ALIGN(0x10000) : { *(.text) } \
# RUN:         . = 0x30000; .data                : { *(.data) } \
# RUN:       }" > %t.script
# RUN: ld.lld %t1.o %t2.o -script %t.script -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex --mcpu=mips32r6 -d -t -s --no-show-raw-insn %t.exe \
# RUN:   | FileCheck %s

  .text
  .globl  __start
__start:
  lwpc      $6, _foo                # R_MIPS_PC19_S2
  beqc      $5, $6, _foo            # R_MIPS_PC16
  beqzc     $9, _foo                # R_MIPS_PC21_S2
  bc        _foo                    # R_MIPS_PC26_S2
  aluipc    $2, %pcrel_hi(_foo)     # R_MIPS_PCHI16
  addiu     $2, $2, %pcrel_lo(_foo) # R_MIPS_PCLO16

  .data
  .word _foo+8-.                    # R_MIPS_PC32

# CHECK: 00020000 g       .text           00000000 __start
# CHECK: 00020020 g       .text           00000000 _foo

# CHECK: Contents of section .data:
# CHECK-NEXT: 30000 ffff0028 00000000 00000000 00000000
#                   ^-- 0x20020 + 8 - 0x30000

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <__start>:
# CHECK-NEXT:    20000:       lwpc    $6, 32
#                                         ^-- (0x20020-0x20000)>>2
# CHECK-NEXT:    20004:       beqc    $5, $6, 0x20020
#                                             ^-- (0x20020-4-0x20004)>>2
# CHECK-NEXT:    20008:       beqzc   $9, 0x20020
#                                         ^-- (0x20020-4-0x20008)>>2
# CHECK-NEXT:    2000c:       bc      0x20020
#                                     ^-- (0x20020-4-0x2000c)>>2
# CHECK-NEXT:    20010:       aluipc  $2, 0
#                                         ^-- %hi(0x20020-0x20010)
# CHECK-NEXT:    20014:       addiu   $2, $2, 12
#                                             ^-- %lo(0x20020-0x20014)
