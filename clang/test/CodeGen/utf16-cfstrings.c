// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s
// <rdar://problem/10655949>

// CHECK: @.str = internal unnamed_addr constant [9 x i16] [i16 252, i16 98, i16 101, i16 114, i16 104, i16 117, i16 110, i16 100, i16 0], align 2

#define CFSTR __builtin___CFStringMakeConstantString

void foo() {
  CFSTR("überhund");
}
