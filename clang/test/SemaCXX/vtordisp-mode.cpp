// RUN: %clang_cc1 -triple i686-pc-win32 -std=c++11 -vtordisp-mode=0 -DVTORDISP_MODE=0 %s -verify
// RUN: %clang_cc1 -triple i686-pc-win32 -std=c++11 -vtordisp-mode=1 -DVTORDISP_MODE=1 %s -verify
// RUN: %clang_cc1 -triple i686-pc-win32 -std=c++11 -vtordisp-mode=2 -DVTORDISP_MODE=2 %s -verify

// expected-no-diagnostics

struct A {
  A();
  virtual void foo();
};

// At /vd1, there is a vtordisp before A.
struct B : virtual A {
  B();
  virtual void foo();
  virtual void bar();
};

// At /vd2, there is a vtordisp before B, but only because it has its own
// vftable.
struct C : virtual B {
  C();
};

// There are two vfptrs, two vbptrs, and some number of vtordisps.
static_assert(sizeof(C) == 2 * 4 + 2 * 4 + 4 * VTORDISP_MODE, "size mismatch");
