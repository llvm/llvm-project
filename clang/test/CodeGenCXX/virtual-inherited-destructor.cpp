// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm-only

struct A { virtual ~A(); };
struct B : A {
  ~B() { }
};
B x;

