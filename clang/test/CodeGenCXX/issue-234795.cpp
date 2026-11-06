// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -O0 -emit-llvm -o - %s | FileCheck %s

struct A {
  int x = 42;
  virtual ~A() {}
};

struct B : virtual A {
  explicit B(int) : B() {}
  B() = default;
};

struct D : B {
  D() : B(0) {}
};

int main() {
  D d;
  B b(0);
  return d.x + b.x;
}

// The complete-object constructor initializes the full object.
// CHECK-LABEL: define{{.*}} @_ZN1BC1Ei
// CHECK: call void @llvm.memset{{.*}}i64 24

// The base-object constructor must not zero-initialize its virtual base.
// CHECK-LABEL: define{{.*}} @_ZN1BC2Ei
// CHECK: call void @llvm.memset{{.*}}i64 8
// CHECK: call{{.*}} @_ZN1BC2Ev