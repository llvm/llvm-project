// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-macho -emit-llvm %s -o %t

// <rdar://problem/10657500>: Check that the backing store of CFStrings are
// constant with the -fwritable-strings flag.
//
// RUN: %clang_cc1 -triple x86_64-macho -fwritable-strings -emit-llvm %s -o - | FileCheck %s
//
// CHECK: @.str = private unnamed_addr constant [14 x i8] c"Hello, World!\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @.str.1 = private unnamed_addr constant [7 x i8] c"yo joe\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @.str.3 = private unnamed_addr constant [16 x i8] c"Goodbye, World!\00", section "__TEXT,__cstring,cstring_literals", align 1

#define CFSTR __builtin___CFStringMakeConstantString

void f(void) {
  CFSTR("Hello, World!");
}

// rdar://6248329
void *G = CFSTR("yo joe");

void h(void) {
  static void* h = CFSTR("Goodbye, World!");
}
