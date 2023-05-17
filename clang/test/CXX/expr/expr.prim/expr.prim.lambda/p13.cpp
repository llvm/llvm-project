// RUN: %clang_cc1 -std=c++11 %s -Wunused -Wno-unused-lambda-capture -Wno-c++14-extensions -verify
// RUN: %clang_cc1 -std=c++17 %s -Wunused -Wno-unused-lambda-capture -Wno-c++14-extensions -verify


const int global = 0;

void f2() {
  int i = 1;
  void g1(int = ([i]{ return i; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g2(int = ([i]{ return 0; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g3(int = ([=]{ return i; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g4(int = ([=]{ return 0; })());
  void g5(int = ([]{ return sizeof i; })());
  void g6(int = ([x=1, y = global, &z = global]{ return x; })());
  void g7(int = ([x=i, &y=i]{ return x; })()); // expected-error 2{{default argument references local variable 'i' of enclosing function}}
}

#if __cplusplus >= 201703L
int global_array[] = { 1, 2 };
auto [ga, gb] = global_array;

void structured_bindings() {
  int array[] = { 1, 2 };
  auto [a, b] = array;
  void func(int c = [x = a, &xref = a, y = ga, &yref = ga] { return x; }()); // expected-error 2{{default argument references local variable 'a' of enclosing function}}
}
#endif

namespace lambda_in_default_args {
  int f(int = [] () -> int { int n; return ++n; } ());
  template<typename T> T g(T = [] () -> T { T n; return ++n; } ());
  int k = f() + g<int>();
}
