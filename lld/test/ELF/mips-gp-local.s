# Check handling of relocations against __gnu_local_gp symbol.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld -o %t.exe %t.o
# RUN: llvm-objdump -d -t %t.exe | FileCheck %s

# REQUIRES: mips

# CHECK:      Disassembly of section .text:
# CHECK-NEXT: __start:
# CHECK-NEXT:    20000:  3c 08 00 00  lui   $8, 0
# CHECK-NEXT:    20004:  21 08 00 00  addi  $8, $8, 0

# CHECK: 00000000  *ABS*  00000000 _gp

  .text
  .globl  __start
__start:
  lui    $t0,%hi(__gnu_local_gp)
  addi   $t0,$t0,%lo(__gnu_local_gp)
