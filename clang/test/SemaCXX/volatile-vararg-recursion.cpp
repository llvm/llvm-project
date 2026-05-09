// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace cwg535 {
class X {
  X(const X &);
};

struct B {
  X y;
  B(...);
};

extern volatile B b1;
B b2(b1); // expected-error {{cannot pass object of non-trivial type 'volatile B' through variadic constructor; call will abort at runtime}}
}
