// RUN: %clang_cc1 -verify -fsyntax-only %s

static int *z[1] = {({ static int _x = 70; &_x; })}; // expected-error {{statement expression not allowed at file scope}}

void T1(void) {
  int *x[1] = {({ static int _x = 10; &_x; })}; // expected-no-error

  /* Before commit
     683e83c5 [Clang][C++2b] P2242R3: Non-literal variables [...] in constexpr
     (i.e in clang-14 and earlier)
     this was silently accepted, but generated incorrect code.
  */
  static int *y[1] = {({ static int _x = 20; &_x; })}; // expected-error {{initializer element is not a compile-time constant}}
}
