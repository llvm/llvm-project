# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386-pc-linux-gnu %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=i386-pc-linux-gnu %S/Inputs/unknown-reloc.s -o %t2.o
# RUN: not ld.lld %t1.o %t2.o -o %t.out 2>&1 | FileCheck %s

# CHECK: do not know how to handle relocation R_386_PC8 (23)
# CHECK: do not know how to handle relocation R_386_8 (22)

.text
.global foo
foo:

.byte und-foo
.byte foo
