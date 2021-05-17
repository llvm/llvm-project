// RUN: %clang_cc1 -verify -std=c++11 %s

[[carries_dependency, carries_dependency]] int m1(); // ok
[[carries_dependency]] [[carries_dependency]] int m2(); // ok
[[carries_dependency()]] int m3(); // expected-error {{attribute 'carries_dependency' cannot have an argument list}}

[[carries_dependency]] void f1(); // FIXME: warn here
[[carries_dependency]] int f2(); // ok
int f3(int param [[carries_dependency]]); // ok
[[carries_dependency]] int (*f4)(); // expected-error {{'carries_dependency' attribute only applies to parameters, Objective-C methods, and functions}}
int (*f5 [[carries_dependency]])(); // expected-error {{'carries_dependency' attribute only applies to}}
int (*f6)() [[carries_dependency]]; // expected-error {{'carries_dependency' attribute cannot be applied to types}}
int (*f7)(int n [[carries_dependency]]); // expected-error {{'[[carries_dependency]]' attribute only allowed on parameter in a function declaration}}
int (((f8)))(int n [[carries_dependency]]); // ok
int (*f9(int n))(int n [[carries_dependency]]); // expected-error {{'[[carries_dependency]]' attribute only allowed on parameter in a function declaration}}
int typedef f10(int n [[carries_dependency]]); // expected-error {{'[[carries_dependency]]' attribute only allowed on parameter in a function declaration}}
using T = int(int n [[carries_dependency]]); // expected-error {{'[[carries_dependency]]' attribute only allowed on parameter in a function declaration}}
struct S {
  [[carries_dependency]] int f(int n [[carries_dependency]]); // ok
  int (*p)(int n [[carries_dependency]]); // expected-error {{'[[carries_dependency]]' attribute only allowed on parameter in a function declaration}}
};
void f() {
  [[carries_dependency]] int f(int n [[carries_dependency]]); // ok
  [[carries_dependency]] // expected-error {{'carries_dependency' attribute only applies to}}
      int (*p)(int n [[carries_dependency]]); // expected-error {{'[[carries_dependency]]' attribute only allowed on parameter in a function declaration}}
}

auto l1 = [](int n [[carries_dependency]]) {};
// There's no way to write a lambda such that the return value carries
// a dependency, because an attribute applied to the lambda appertains to
// the *type* of the operator() function, not to the function itself.
auto l2 = []() [[carries_dependency]] {}; // expected-error {{'carries_dependency' attribute cannot be applied to types}}
