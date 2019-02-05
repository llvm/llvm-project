// RUN: %clang_cc1 %s -verify -DDIAG1
// RUN: %clang_cc1 %s -verify -DDIAG1 -DDIAG2 -Wdelete-non-virtual-dtor
// RUN: %clang_cc1 %s -verify -DDIAG1         -Wmost -Wno-delete-non-abstract-non-virtual-dtor
// RUN: %clang_cc1 %s -verify         -DDIAG2 -Wmost -Wno-delete-abstract-non-virtual-dtor
// RUN: %clang_cc1 %s -verify                 -Wmost -Wno-delete-non-virtual-dtor

#ifndef DIAG1
#ifndef DIAG2
// expected-no-diagnostics
#endif
#endif

struct S1 {
  ~S1() {}
  virtual void abs() = 0;
};

void f1(S1 *s1) { delete s1; }
#ifdef DIAG1
// expected-warning@-2 {{delete called on 'S1' that is abstract but has non-virtual destructor}}
#endif

struct S2 {
  ~S2() {}
  virtual void real() {}
};
void f2(S2 *s2) { delete s2; }
#ifdef DIAG2
// expected-warning@-2 {{delete called on non-final 'S2' that has virtual functions but non-virtual destructor}}
#endif
