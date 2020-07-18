// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

void h() {
  int i1 = 0;
  extern void h1(int x = i1);
  // expected-error@-1 {{default argument references local variable 'i1' of enclosing function}}

  const int i2 = 0;
  extern void h2a(int x = i2);     // ok, not odr-use
  extern void h2b(int x = i2 + 0); // ok, not odr-use

  const int i3 = 0;
  extern void h3(const int *x = &i3);
  // expected-error@-1 {{default argument references local variable 'i3' of enclosing function}}

  const int i4 = 0;
  extern void h4(int x = sizeof(i4));         // ok, not odr-use
  extern void h5(int x = decltype(i4 + 4)()); // ok, not odr-use

  union {
    int i5;
  };

  extern void h6(int = i5);
  // expected-error@-1 {{default argument references local variable '' of enclosing function}}

  struct S { int i; };
  auto [x] = S();

  extern void h7(int = x); // FIXME: reject
}
