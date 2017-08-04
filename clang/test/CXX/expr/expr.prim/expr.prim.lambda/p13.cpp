// RUN: %clang_cc1 -std=c++11 %s -Wunused -Wno-unused-lambda-capture -verify

void f2() {
  int i = 1;
  void g1(int = ([i]{ return i; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g2(int = ([i]{ return 0; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g3(int = ([=]{ return i; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g4(int = ([=]{ return 0; })());
  void g5(int = ([]{ return sizeof i; })());
}

namespace lambda_in_default_args {
  int f(int = [] () -> int { int n; return ++n; } ());
  template<typename T> T g(T = [] () -> T { T n; return ++n; } ());
  int k = f() + g<int>();
}
