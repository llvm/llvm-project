// RUN: %clang_cc1 -std=c++2a -verify %s

constinit int a;
constinit thread_local int b;
constinit static int c;

void f() {
  constinit static int a;
  constinit thread_local int b;
  constinit int c; // expected-error {{local variable cannot be declared 'constinit'}}
}

namespace missing {
  int a; // expected-note {{add the 'constinit' specifier}}
  extern constinit int a; // expected-error {{added after initialization}}

  // We allow inheriting 'constinit' from a forward declaration as an extension.
  extern constinit int b; // expected-note {{here}}
  int b; // expected-warning {{'constinit' specifier missing}}
}

struct S {
  static constinit int a; // expected-note {{here}}
  static constinit constexpr int b; // expected-error {{cannot combine with previous}} expected-note {{here}}
  static constinit const int c = 1;
  static constinit const int d = 1;
};
int S::a; // expected-warning {{'constinit' specifier missing}}
int S::b; // expected-warning {{'constinit' specifier missing}}
const int S::c;
inline const int S::d;

struct T {
  static int a;
  static constexpr int b = 1; // expected-note {{add the 'constinit' specifier}}
  static const int c = 1; // expected-note {{add the 'constinit' specifier}}
  static const int d = 1; // expected-note {{add the 'constinit' specifier}}
};
constinit int T::a;
constinit const int T::b; // expected-error {{'constinit' specifier added after initialization}}
constinit const int T::c; // expected-error {{'constinit' specifier added after initialization}}
constinit inline const int T::d; // expected-error {{'constinit' specifier added after initialization}}

constinit void g() {} // expected-error {{constinit can only be used in variable declarations}}

// (These used to trigger crashes.)
void h();
constinit void h(); // expected-error {{constinit can only be used in variable declarations}}
constexpr void i(); // expected-note {{here}}
constinit void i(); // expected-error {{non-constexpr declaration of 'i' follows constexpr declaration}}
// expected-error@-1 {{constinit can only be used in variable declarations}} 

typedef constinit int type; // expected-error {{typedef cannot be constinit}}
using type = constinit int; // expected-error {{type name does not allow constinit specifier}}
auto q() -> int constinit; // expected-error {{type name does not allow constinit specifier}}
