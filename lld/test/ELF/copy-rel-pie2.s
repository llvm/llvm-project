// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-mc %p/Inputs/copy-rel-pie.s -o %t2.o -filetype=obj -triple=x86_64-pc-linux
// RUN: ld.lld %t2.o -o %t2.so -shared
// RUN: ld.lld %t.o %t2.so -o %t -pie
// RUN: llvm-readobj -r %t | FileCheck %s

// CHECK: R_X86_64_COPY
// CHECK: R_X86_64_JUMP_SLOT

.rodata
.quad bar
.quad foo
