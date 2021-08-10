# REQUIRES: amdgpu
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=amdgcn--amdhsa -mcpu=fiji %t/asm -o %t.o
# RUN: ld.lld %t.o -o %t/out --script %t/script
# RUN: llvm-objdump -d %t/out | FileCheck %s


#--- script
SECTIONS {
  . = 0x1000;
  .text.likely : { *(.text.likely) }
  . = 0x2000;
  .text : { *(.text) }
  . = 0x3000;
  .text.unlikely : { *(.text.unlikely) }
}


#--- asm
.section .text.likely
hot1:
  s_add_i32 s15, s15, 1
hot2:
  s_add_i32 s13, s13, 1
.text
foo:
  s_branch cold2
  s_branch hot2
.section .text.unlikely
cold1:
  s_add_i32 s15, s15, 1
  s_add_i32 s14, s14, 1
cold2:
  s_add_i32 s13, s13, 1

# CHECK:  <foo>
# CHECK-NEXT: s_branch 1025
# CHECK-NEXT: s_branch 64511
