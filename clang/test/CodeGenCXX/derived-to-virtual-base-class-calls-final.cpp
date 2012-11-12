// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

struct A { int i; };
struct B { char j; };
struct C : A, B { int k; };

struct D final : virtual C { 
  D(); 
  virtual void f();
};

// CHECK: define %struct.B* @_Z1fR1D
B &f(D &d) {
  // CHECK-NOT: load i8**
  return d;
}
