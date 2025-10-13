// RUN: %clang_cc1 -fsyntax-only -std=c++98 -verify %s

namespace N2 {
  struct S1;

  namespace N1 {
    class C1 {};

    struct S2 { // expected-note {{defined here}}
                // expected-note@-1 {{defined here}}
                // expected-note@-2 {{defined here}}
                // expected-note@-3 {{defined here}}
                // expected-note@-4 {{defined here}}
      void func(S1*); // expected-note {{type of 1st parameter of member declaration does not match definition ('S1 *' (aka 'N2::S1 *') vs 'S1 *' (aka 'N2::N1::S1 *'))}}
      void func(C1&, unsigned, const S1*); // expected-note {{type of 3rd parameter of member declaration does not match definition ('const S1 *' (aka 'const N2::S1 *') vs 'const S1 *' (aka 'const N2::N1::S1 *'))}}
      void func(const S1*, unsigned); //expected-note {{type of 1st parameter of member declaration does not match definition ('const S1 *' vs 'S1')}}
      void func(unsigned, const S1*); // expected-note {{type of 1st parameter of member declaration does not match definition ('unsigned int' vs 'unsigned int *')}}
    };

    struct S1 {};
  }
}

void N2::N1::S2::func(S1*) {} // expected-error {{out-of-line definition of 'func' does not match any declaration in 'N2::N1::S2'}}
void N2::N1::S2::func(C1&, unsigned, const S1*) {} // expected-error {{out-of-line definition of 'func' does not match any declaration in 'N2::N1::S2'}}
void N2::N1::S2::func(S1*, double) {} // expected-error {{out-of-line definition of 'func' does not match any declaration in 'N2::N1::S2'}}
void N2::N1::S2::func(S1, unsigned) {} // expected-error {{out-of-line definition of 'func' does not match any declaration in 'N2::N1::S2'}}
void N2::N1::S2::func(unsigned*, S1*) {} // expected-error {{out-of-line definition of 'func' does not match any declaration in 'N2::N1::S2'}}
