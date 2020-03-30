// RUN: %clang_cc1 -fobjc-runtime=gcc -frecovery-ast -verify %s
// RUN: %clang_cc1 -fobjc-runtime=gcc -fno-recovery-ast -verify %s

@interface Ivar
{
  int Foo : foo(); // expected-error {{use of undeclared identifier}}
};
@end

struct X { int Y: foo(); }; // expected-error {{use of undeclared identifier}}

constexpr int s = sizeof(Ivar);
constexpr int ss = sizeof(X);
