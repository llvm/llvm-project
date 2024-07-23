// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s


int arr[2];
// CHECK: @pastEnd = constant ptr getelementptr (i8, ptr @arr, i64 8)
int &pastEnd = arr[2];

// CHECK: @F = constant ptr @arr, align 8
int &F = arr[0];

struct S {
  int a;
  float c[3];
};

// CHECK: @s = global %struct.S zeroinitializer, align 4
S s;
// CHECK: @sp = constant ptr getelementptr (i8, ptr @s, i64 16), align 8
float &sp = s.c[3];


namespace BaseClassOffsets {
  struct A { int a; };
  struct B { int b; };
  struct C : A, B { int c; };

  extern C c;
  // CHECK: @_ZN16BaseClassOffsets1aE = global ptr @_ZN16BaseClassOffsets1cE, align 8
  A* a = &c;
  // CHECK: @_ZN16BaseClassOffsets1bE = global ptr getelementptr (i8, ptr @_ZN16BaseClassOffsets1cE, i64 4), align 8
  B* b = &c;
}
