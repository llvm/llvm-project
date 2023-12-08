// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
  namespace B { }
}

using A::B; // expected-error{{using declaration cannot refer to a namespace}}
