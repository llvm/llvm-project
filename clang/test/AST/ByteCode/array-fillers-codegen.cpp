// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux -fexperimental-new-constant-interpreter -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux                                         -emit-llvm -o - %s | FileCheck %s

struct S {
  int a;
  int b;
};

// CHECK: @GA = global <{ %struct.S, [99 x %struct.S] }> <{ %struct.S { i32 1, i32 2 }, [99 x %struct.S] zeroinitializer }>
S GA[100] = {{1, 2}};
// CHECK: @GP = global ptr getelementptr (i8, ptr @GA, i64 484)
int *GP = &GA[60].b;

constexpr int A[100] = {1, 2};
// CHECK: @_ZL1A = internal constant <{ i32, i32, [98 x i32] }> <{ i32 1, i32 2, [98 x i32] zeroinitializer }>
const int *UseA() { return A; }

