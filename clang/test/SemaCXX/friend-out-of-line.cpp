// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace N {
  class X;
};

class N::X {
  template<typename T> friend const T& f(const X&);
  friend const int& g(const X&);
  friend class Y;
};
