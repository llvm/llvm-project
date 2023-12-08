// RUN: %clang_cc1 -fsyntax-only -verify %s

void h() {
  void f1(int x, int y = sizeof(x));      // ok
  void f2(int x, int y = decltype(x)());  // ok
  void f3(int x, int y = x);
  // expected-error@-1 {{default argument references parameter 'x'}}
  void f4(int x, int y = x + 0);
  // expected-error@-1 {{default argument references parameter 'x'}}
  void f5(int x, int y = ((void)x, 0));
  // expected-error@-1 {{default argument references parameter 'x'}}
}
