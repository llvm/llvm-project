# Check warning on orphaned R_MIPS_HI16 relocations.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe 2>&1 | FileCheck -check-prefix=WARN %s
# RUN: llvm-objdump -d -t %t.exe | FileCheck %s

# REQUIRES: mips

  .text
  .globl  __start
__start:
  lui    $t0,%hi(__start+0x10000)

# WARN: Can't find matching R_MIPS_LO16 relocation for R_MIPS_HI16

# CHECK:      Disassembly of section .text:
# CHECK-NEXT: __start:
# CHECK-NEXT:  20000:   3c 08 00 02   lui    $8, 2
#                                                ^-- %hi(__start) w/o addend

# CHECK: SYMBOL TABLE:
# CHECK: 0020000     .text   00000000 __start
