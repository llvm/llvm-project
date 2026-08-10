// RUN: %clang_cc1 %s -triple i386-unknown-unknown -emit-llvm -o - | FileCheck %s

struct A {
  virtual int f() { return 1; }
};

struct B {
  virtual int f() { return 2; }
};

struct C : A, B {
  virtual int f() { return 3; }
};

struct D : C {
  virtual int f() { return 4; }
};

static int f(D* d) {
  B* b = d;
  return b->f();
};

int g() {
  D d;
  return f(&d);
}

// Thunks should be marked as "linkonce ODR" not "weak".
//
// CHECK: define linkonce_odr noundef i32 @_ZThn{{[48]}}_N1D1fEv
// CHECK: define linkonce_odr noundef i32 @_ZThn{{[48]}}_N1C1fEv
