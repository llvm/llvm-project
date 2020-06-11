// RUN: %clang_cc1 -fsyntax-only -verify %s

void h() {
  int i1 = 0;
  extern void h1(int x = i1);
  // expected-error@-1 {{default argument references local variable 'i1' of enclosing function}}

  const int i2 = 0;
  extern void h2a(int x = i2); // FIXME: ok, not odr-use
  // expected-error@-1 {{default argument references local variable 'i2' of enclosing function}}
  extern void h2b(int x = i2 + 0); // ok, not odr-use

  const int i3 = 0;
  extern void h3(const int *x = &i3);
  // expected-error@-1 {{default argument references local variable 'i3' of enclosing function}}

  const int i4 = 0;
  extern void h4(int x = sizeof(i4));         // ok, not odr-use
  extern void h5(int x = decltype(i4 + 4)()); // ok, not odr-use
}
