// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
  struct x {}; // expected-note {{candidate found by name lookup is 'A::x'}}
  int x; // expected-note {{candidate found by name lookup is 'A::x'}}

  struct y {}; // expected-note {{type declaration hidden}}

  struct z;
  void z(float);
}

namespace B {
  struct x {}; // expected-note {{candidate found by name lookup is 'B::x'}}
  float x; // expected-note {{candidate found by name lookup is 'B::x'}}

  float y; // expected-note {{declaration hides type}}

  void z(int);
}

namespace AB {
  using namespace A;
  using namespace B;
}

void test() {
  struct AB::x foo; // expected-error {{reference to 'x' is ambiguous}}
  int i = AB::x; // expected-error {{reference to 'x' is ambiguous}}

  struct AB::y bar;
  float f = AB::y; // expected-error {{a type named 'y' is hidden by a declaration in a different namespace}}
  AB::z(i);
  AB::z(f);
}
