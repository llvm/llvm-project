// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -Wno-unused-value -emit-llvm -o - %s | FileCheck %s

struct A {
  int &x;
  ~A() { x = 0; }
};

struct AA {
  int &x;
  ~AA() { x = -1; }
};

struct B {
  int &x;
  const A &a = A{x};
};

struct BB {
  int &x;
  const AA &a = AA{x};
};

// CHECK-LABEL: define{{.*}} i32 @_Z3onev()
int one() {
  int x = 1;
  B{x};
  // CHECK: call void @_ZN1AD{{[012]}}Ev
  // CHECK-NEXT: load i32, ptr
  return x;
}

// The default initializers are part of the same full-expression, so their
// temporaries are destroyed in reverse construction order.
// CHECK-LABEL: define{{.*}} i32 @_Z3twov()
int two() {
  int x = 1;
  B{x}, BB{x};
  // CHECK: call void @_ZN2AAD{{[012]}}Ev
  // CHECK-NEXT: call void @_ZN1AD{{[012]}}Ev
  // CHECK-NEXT: load i32, ptr
  return x;
}
