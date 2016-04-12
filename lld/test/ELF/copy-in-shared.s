// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/copy-in-shared.s -o %t1.o
// RUN: ld.lld -shared %t1.o -o %t1.so
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t2.o
// RUN: ld.lld %t2.o %t1.so -o %t2.so -shared
// RUN: llvm-readobj -r %t2.so | FileCheck %s
// REQUIRES: x86

.quad foo

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     R_X86_64_64 foo 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
