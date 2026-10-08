// UNSUPPORTED: target={{.*}}-zos{{.*}}
// RUN: split-file %s %t
//--- foo.h
//--- tu.c
asm(".incbin \"foo.h\"");
// RUN: cd %t
// RUN: %clang -c -emit-llvm %t/tu.c -o %t/tu.ll
// RUN: llvm-dis %t/tu.ll -o - | FileCheck %s
// CHECK: module asm
// CHECK-NEXT: ".incbin \22foo.h\22"
