# Check R_MIPS_HI16 / LO16 relocations calculation against _gp_disp.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-dynamic.s -o %t2.o
# RUN: ld.lld %t1.o %t2.o -o %t.exe
# RUN: llvm-objdump -d -t %t.exe | FileCheck -check-prefix=EXE %s
# RUN: ld.lld %t1.o %t2.o -shared -o %t.so
# RUN: llvm-objdump -d -t %t.so | FileCheck -check-prefix=SO %s

# REQUIRES: mips

  .text
  .globl  __start
__start:
  lui    $t0,%hi(_gp_disp)
  addi   $t0,$t0,%lo(_gp_disp)
  lw     $v0,%call16(_foo)($gp)

# EXE:      Disassembly of section .text:
# EXE-NEXT: __start:
# EXE-NEXT:  20000:   3c 08 00 01   lui    $8, 1
#                                              ^-- %hi(0x37ff0-0x20000)
# EXE-NEXT:  20004:   21 08 7f f0   addi   $8, $8, 32752
#                                                  ^-- %lo(0x37ff0-0x20004+4)

# EXE: SYMBOL TABLE:
# EXE: 00037ff0     *ABS*   00000000 _gp
# EXE: 00020000     .text   00000000 __start
# EXE: 00020010     .text   00000000 _foo

# SO:      Disassembly of section .text:
# SO-NEXT: __start:
# SO-NEXT:  10000:   3c 08 00 01   lui    $8, 1
#                                             ^-- %hi(0x27ff0-0x10000)
# SO-NEXT:  10004:   21 08 7f f0   addi   $8, $8, 32752
#                                                 ^-- %lo(0x27ff0-0x10004+4)

# SO: SYMBOL TABLE:
# SO: 00027ff0     *ABS*   00000000 _gp
# SO: 00010000     .text   00000000 __start
# SO: 00010010     .text   00000000 _foo
