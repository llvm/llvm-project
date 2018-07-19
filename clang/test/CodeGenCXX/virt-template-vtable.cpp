// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

template<class T> class A {
public:
  A() {}
  virtual void a() {}
};
class B : A<int> {
  B();
};
B::B() {}

template class A<long>;

extern template class A<short>;
template class A<short>;


// CHECK: @_ZTV1B = linkonce_odr unnamed_addr constant
// CHECK: @_ZTV1AIlE = weak_odr unnamed_addr constant
// CHECK: @_ZTV1AIsE = weak_odr unnamed_addr constant
// CHECK: @_ZTV1AIiE = linkonce_odr unnamed_addr constant

template<class T> struct C {
  virtual void c() {}
};
struct D : C<int> {
  virtual void d();
};
void D::d() {}

// CHECK: define {{.*}}@_ZN1CIiE1cEv(
