// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// END.
# 1 "t.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "t.c"
int __attribute((annotate("foo"))) foo(void) { return 0; }

// CHECK: private unnamed_addr constant [4 x i8] c"t.c\00"
// CHECK: @llvm.global.annotations = {{.*}}, i32 1 }
