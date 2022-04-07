// RUN: %clang_cc1 -no-opaque-pointers %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

struct A { int i; };
struct B { char j; };
struct C : A, B { int k; };

struct D final : virtual C { 
  D(); 
  virtual void f();
};

// CHECK-LABEL: define {{.*}}dereferenceable({{[0-9]+}}) %struct.B* @_Z1fR1D
B &f(D &d) {
  // CHECK-NOT: load i8*, i8**
  return d;
}
