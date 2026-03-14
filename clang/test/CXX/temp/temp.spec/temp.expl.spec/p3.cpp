// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace N {
  template<class T> class X; // expected-note {{'N::X' declared here}}
}

template<> class X<int> { /* ... */ };	// expected-error {{no template named 'X'; did you mean 'N::X'?}}

namespace N {
  
template<> class X<char*> { /* ... */ };	// OK: X is a template
  
}
