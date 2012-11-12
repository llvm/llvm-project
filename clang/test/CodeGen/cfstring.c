// RUN: %clang_cc1 -emit-llvm %s -o %t

// <rdar://problem/10657500>: Check that the backing store of CFStrings are
// constant with the -fwritable-strings flag.
//
// RUN: %clang_cc1 -fwritable-strings -emit-llvm %s -o - | FileCheck %s
//
// CHECK: @.str = linker_private unnamed_addr constant [14 x i8] c"Hello, World!\00", align 1
// CHECK: @.str1 = linker_private unnamed_addr constant [7 x i8] c"yo joe\00", align 1
// CHECK: @.str3 = linker_private unnamed_addr constant [16 x i8] c"Goodbye, World!\00", align 1

#define CFSTR __builtin___CFStringMakeConstantString

void f() {
  CFSTR("Hello, World!");
}

// rdar://6248329
void *G = CFSTR("yo joe");

void h() {
  static void* h = CFSTR("Goodbye, World!");
}
