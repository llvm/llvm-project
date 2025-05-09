// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-uninitialized  -verify=good %s
//good-no-diagnostics

template <class T>
class RefMem { // expected-warning {{class 'RefMem<int &>' does not declare any constructor to initialize its non-modifiable members}}
  T &M; // expected-note {{reference member 'M' will never be initialized}}
};

struct RefRef {
  RefMem<int &> R; // expected-note {{in instantiation of template class 'RefMem<int &>' requested here}}
};

