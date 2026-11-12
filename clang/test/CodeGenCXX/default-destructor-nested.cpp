// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm-only
// PR6294

class A {
public: virtual ~A();
};
class B {
  class C;
};
class B::C : public A {
  C();
};
B::C::C() {}
