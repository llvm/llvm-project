# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "SECTIONS { .foo : { *(.foo.*) } }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -s %t1 | FileCheck %s
# CHECK:      Contents of section .foo:
# CHECK-NEXT:  0158 01000000 02000000 00000000 73686f72  ............shor
# CHECK-NEXT:  0168 7420756e 7369676e 65642069 6e7400    t unsigned int.

.global _start
_start:
  nop

.section .foo.1, "aw"
writable:
 .long 1

.section .foo.2, "aM",@progbits,1
readable:
 .long 2

.section .foo.3, "awx"
 .long 0

.section .foo.4, "MS",@progbits,1
.LASF2:
 .string "short unsigned int"
