# REQUIRES: mips
# Check R_MIPS_GOT16 relocation against merge section.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -o %t.o %s
# RUN: ld.lld -shared -o %t.so %t.o
# RUN: llvm-objdump -dD %t.so | FileCheck %s

# CHECK: 000001f1 .rodata:
#                'f''o''o''\0'
# CHECK-NEXT: 1f1: 66 6f 6f 00
# CHECK: lw      $25, -32744($gp)
#                            0x1f1
# CHECK-NEXT: addiu   $4, $25, 497

  .text
  lw     $t9, %got($.str)($gp)
  addiu  $a0, $t9, %lo($.str)

  .section  .rodata.str,"aMS",@progbits,1
$.str:
  .asciz "foo"
