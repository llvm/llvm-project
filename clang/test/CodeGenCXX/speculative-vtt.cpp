// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -O2 -disable-llvm-passes -emit-llvm -o - | FileCheck %s
struct A { virtual ~A(); };
template<typename T> struct B : virtual A {
  ~B() override {}
};
struct C : B<int>, B<float> { C(); ~C() override; };
struct D : C { ~D() override; };

// We must not create a reference to B<int>::~B() here, because we're not going to emit it.
// CHECK-NOT: @_ZN1BIiED1Ev
// CHECK-NOT: @_ZTC1D0_1BIiE =
// CHECK-NOT: @_ZTT1D = available_externally
D *p = new D;
