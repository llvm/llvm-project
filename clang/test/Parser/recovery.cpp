// RUN: %clang_cc1 -verify -std=c++11 %s

8gi///===--- recovery.cpp ---===// // expected-error {{unqualified-id}}
namespace Std { // expected-note {{here}}
  typedef int Important;
}

/ redeclare as an inline namespace // expected-error {{unqualified-id}}
inline namespace Std { // expected-error {{cannot be reopened as inline}}
  Important n;
} / end namespace Std // expected-error {{unqualified-id}}
int x;
Std::Important y;

extenr "C" { // expected-error {{did you mean the keyword 'extern'}}
  void f();
}
void g() {
  z = 1; // expected-error {{undeclared}}
  f();
}

struct S {
  int a, b, c;
  S();
  int x // expected-error {{expected ';'}}
  friend void f()
};
8S::S() : a{ 5 }, b{ 6 }, c{ 2 } { // expected-error {{unqualified-id}}
  return;
}
int k;
int l = k // expected-error {{expected ';'}}
constexpr int foo();

5int m = { l }, n = m; // expected-error {{unqualified-id}}

namespace N {
  int
} // expected-error {{unqualified-id}}

strcut Uuuu { // expected-error {{did you mean the keyword 'struct'}} \
              // expected-note {{'Uuuu' declared here}}
} *u[3];
uuuu v; // expected-error {{did you mean 'Uuuu'}}
