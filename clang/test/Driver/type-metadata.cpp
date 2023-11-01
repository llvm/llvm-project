// Test that type metadata are emitted with -fprofile-generate
//
// RUN: %clang -fprofile-generate -fno-lto -target x86_64-unknown-linux -emit-llvm -S %s -o - | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang -fprofile-generate -fno-lto -target x86_64-pc-windows-msvc -emit-llvm -S %s -o - | FileCheck %s --check-prefix=MS

// Test that type metadata are emitted with -fprofile-use
//
// RUN: llvm-profdata merge %S/Inputs/irpgo.proftext -o %t-ir.profdata
// RUN: %clang -fprofile-use=%t-ir.profdata -fno-lto -target x86_64-unknown-linux -emit-llvm -S %s -o - | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang -fprofile-use=%t-ir.profdata -fno-lto -target x86_64-pc-windows-msvc -emit-llvm -S %s -o - | FileCheck %s --check-prefix=MS

// ITANIUM: @_ZTV1A = {{[^!]*}}, !type [[A16:![0-9]+]]
// ITANIUM-SAME:  !type [[AF16:![0-9]+]]

// ITANIUM: @_ZTV1B = {{[^!]*}}, !type  [[A32:![0-9]+]]
// ITANIUM-SAME: !type [[AF32:![0-9]+]]
// ITANIUM-SAME: !type [[AF40:![0-9]+]]
// ITANIUM-SAME: !type [[AF48:![0-9]+]]
// ITANIUM-SAME: !type [[B32:![0-9]+]]
// ITANIUM-SAME: !type [[BF32:![0-9]+]]
// ITANIUM-SAME: !type [[BF40:![0-9]+]]
// ITANIUM-SAME: !type [[BF48:![0-9]+]]

// ITANIUM: @_ZTV1C = {{[^!]*}}, !type [[A32]]
// ITANIUM-SAME: !type [[AF32]]
// ITANIUM-SAME: !type [[C32:![0-9]+]]
// ITANIUM-SAME: !type [[CF32:![0-9]+]]

// ITANIUM: @_ZTVN12_GLOBAL__N_11DE = {{[^!]*}}, !type [[A32]]
// ITANIUM-SAME: !type [[AF32]]
// ITANIUM-SAME: !type [[AF40]]
// ITANIUM-SAME: !type [[AF48]]
// ITANIUM-SAME: !type [[B32]]
// ITANIUM-SAME: !type [[BF32]]
// ITANIUM-SAME: !type [[BF40]]
// ITANIUM-SAME: !type [[BF48]]
// ITANIUM-SAME: !type [[C88:![0-9]+]]
// ITANIUM-SAME: !type [[CF32]]
// ITANIUM-SAME: !type [[CF40:![0-9]+]]
// ITANIUM-SAME: !type [[CF48:![0-9]+]]
// ITANIUM-SAME: !type [[D32:![0-9]+]]
// ITANIUM-SAME: !type [[DF32:![0-9]+]]
// ITANIUM-SAME: !type [[DF40:![0-9]+]]
// ITANIUM-SAME: !type [[DF48:![0-9]+]]

// ITANIUM: @_ZTCN12_GLOBAL__N_11DE0_1B = {{[^!]*}}, !type [[A32]]
// ITANIUM-SAME: !type [[B32]]

// ITANIUM: @_ZTCN12_GLOBAL__N_11DE8_1C = {{[^!]*}}, !type [[A64:![0-9]+]]
// ITANIUM-SAME: !type [[AF64:![0-9]+]]
// ITANIUM-SAME: !type [[C32]]
// ITANIUM-SAME: !type [[CF64:![0-9]+]]

// ITANIUM: @_ZTVZ3foovE2FA = {{[^!]*}}, !type [[A16]]
// ITANIUM-SAME: !type [[AF16]]
// ITANIUM-SAME: !type [[FA16:![0-9]+]]
// ITANIUM-SAME: !type [[FAF16:![0-9]+]]

// MS: comdat($"??_7A@@6B@"), !type [[A8:![0-9]+]]
// MS: comdat($"??_7B@@6B0@@"), !type [[B8:![0-9]+]]
// MS: comdat($"??_7B@@6BA@@@"), !type [[A8]]
// MS: comdat($"??_7C@@6B@"), !type [[A8]]
// MS: comdat($"??_7D@?A0x{{[^@]*}}@@6BB@@@"), !type [[B8]], !type [[D8:![0-9]+]]
// MS: comdat($"??_7D@?A0x{{[^@]*}}@@6BA@@@"), !type [[A8]]
// MS: comdat($"??_7FA@?1??foo@@YAXXZ@6B@"), !type [[A8]], !type [[FA8:![0-9]+]]

struct A {
  A();
  virtual void f();
};

struct B : virtual A {
  B();
  virtual void g();
  virtual void h();
};

struct C : virtual A {
  C();
};

namespace {

struct D : B, C {
  D();
  virtual void f();
  virtual void h();
};

}

A::A() {}
B::B() {}
C::C() {}
D::D() {}

void A::f() {
}

void B::g() {
}

void D::f() {
}

void D::h() {
}

void af(A *a) {
  a->f();
}

void df1(D *d) {
 
  d->f();
}

void dg1(D *d) {
 
  d->g();
}

void dh1(D *d) {
  
  d->h();
}

void df2(D *d) {

  d->f();
}

void df3(D *d) {

  d->f();
}

D d;

void foo() {
  df1(&d);
  dg1(&d);
  dh1(&d);
  df2(&d);
  df3(&d);

  struct FA : A {
    void f() {}
  } fa;
  af(&fa);
}
