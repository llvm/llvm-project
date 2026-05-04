// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace GH143216 {
#define A x y
enum { A }; // expected-error {{missing ',' between enumerators}}

#define B x y
void f() {
    int a[2];
    auto [B] = a; // expected-error {{expected ','}}
}

#define C <int!
template <class T> class D;
D C; // expected-error {{expected unqualified-id}} \
     // expected-error {{expected '>'}} \
     // expected-note {{to match this '<'}}

#define E F::{
class F { E }}; // expected-error {{expected identifier}} \
                // expected-error {{expected member name or ';' after declaration specifiers}}
}
