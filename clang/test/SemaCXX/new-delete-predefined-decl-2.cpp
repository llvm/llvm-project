// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -DQUALIFIED -fsyntax-only -verify %s
// expected-no-diagnostics

// PR5904
void f0(int *ptr) {
#ifndef QUALIFIED
  operator delete(ptr);
#endif
}

void f1(int *ptr) {
  ::operator delete[](ptr);
}
