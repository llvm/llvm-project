// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace GH143216 {
#define A x y // expected-error {{missing ',' between enumerators}}
enum { A };

#define B x y // expected-error {{expected ','}}
void f() {
    int a[2];
    auto [B] = a;
}

#define C <int! // expected-error {{expected '>'}}
template <class T> class D;
D C; // expected-error {{expected unqualified-id}} \
     // expected-note {{to match this '<'}}

#define E F::{  // expected-error {{expected identifier}}
class F { E }}; // expected-error {{expected member name or ';' after declaration specifiers}}
}
