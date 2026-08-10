// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// C++0x [class.nest] p3:
//   If class X is defined in a namespace scope, a nested class Y may be
//   declared in class X and later defined in the definition of class X or be
//   later defined in a namespace scope enclosing the definition of class X.

namespace example {
  class E {
    class I1;
    class I2;
    class I1 { };
  };
  class E::I2 { };
}

// Don't insert out-of-line inner class definitions into the namespace scope.
namespace PR6107 {
  struct S1 { };
  struct S2 {
    struct S1;
  };
  struct S2::S1 { };
  S1 s1;
}
