// RUN: %clang_cc1 -emit-llvm-only -triple %itanium_abi_triple %s -emit-llvm -o - %s | FileCheck %s

struct A {};
struct alignas(32) B : virtual A {
  char c[32];
};
struct Pad {
  char c[7];
};
struct C : B, Pad, virtual A {};

struct X {
  virtual A &f();
};
struct U {
  virtual ~U();
};
C c;
struct Y : U, X {
  virtual B &f() override { return c; }
};

Y y;

// FIXME: The return type should be  align 1 dereferenceable(1) %{{[^*]+}}*
// CHECK: define linkonce_odr %{{[^*]+}}* @_ZTchn8_v0_n24_N1Y1fEv(%{{[^*]+}}* %this) unnamed_addr #1 comdat align 2 {
