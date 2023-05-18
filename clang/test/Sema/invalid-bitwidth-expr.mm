// RUN: %clang_cc1 -fobjc-runtime=gcc -frecovery-ast -verify %s

@interface Ivar
{
  int Foo : foo(); // expected-error {{use of undeclared identifier}}
};
@end

struct X { int Y: foo(); }; // expected-error {{use of undeclared identifier}}

constexpr int s = sizeof(Ivar);
constexpr int ss = sizeof(X);

auto func() {
  return undef(); // expected-error {{use of undeclared identifier}}
}
struct Y {
  int X : func();
};
constexpr int sss = sizeof(Y);

bool Foo(int *); // expected-note {{candidate function not viable}}
template <typename T>
struct Base {};
template <typename T>
auto func() {
  // error-bit should be propagated from TemplateArgument to NestNameSpecifier.
  class Base<decltype(Foo(T()))>::type C; // expected-error {{no matching function for call to 'Foo'}}
  return C;
}
struct Z {
  int X : func<int>(); // expected-note {{in instantiation of function template}}
};
constexpr int ssss = sizeof(Z);

struct Z2 {
  int X : sizeof(_BitInt(invalid())); // expected-error {{use of undeclared identifier}}
};
constexpr int sssss = sizeof(Z2);
