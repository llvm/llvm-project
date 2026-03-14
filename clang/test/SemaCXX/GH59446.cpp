// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace GH59446 { // expected-note {{to match this '{'}}
namespace N {
    template <typename T> struct X ; // expected-note 2 {{template is declared here}}
                                     // expected-note@-1 {{'N::X' declared here}}
				     // expected-note@-2 {{non-type declaration found by destructor name lookup}}
  }
  void f(X<int> *x) { // expected-error {{no template named 'X'; did you mean 'N::X'}}
    x->N::X<int>::~X(); // expected-error 2 {{implicit instantiation of undefined template 'GH59446::N::X<int>'}}
                        // expected-error@-1 {{identifier 'X' after '~' in destructor name does not name a type}}
} // expected-error {{expected '}'}}
