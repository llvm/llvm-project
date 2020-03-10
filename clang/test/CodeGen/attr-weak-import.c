// RUN: %clang_cc1 -triple x86_64-darwin-apple -emit-llvm -o - %s | FileCheck %s
// rdar://9538608

extern int A __attribute__((weak_import));
int A;

extern int B __attribute__((weak_import));
extern int B;

int C;
extern int C __attribute__((weak_import));

extern int D __attribute__((weak_import));
extern int D __attribute__((weak_import));
int D;

extern int E __attribute__((weak_import));
int E;
extern int E __attribute__((weak_import));

// CHECK: @A = global i32
// CHECK-NOT: @B =
// CHECK: @C = global i32
// CHECK: @D = global i32
// CHECK: @E = global i32

