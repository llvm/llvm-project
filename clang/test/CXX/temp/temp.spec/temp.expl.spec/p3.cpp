// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace N {
  template<class T> class X; // expected-note {{'N::X' declared here}}
#if __cplusplus <= 199711L
  // expected-note@-2 {{explicitly specialized declaration is here}}
#endif
}

// TODO: Don't add a namespace qualifier to the template if it would trigger
// the warning about the specialization being outside of the namespace.
template<> class X<int> { /* ... */ };	// expected-error {{no template named 'X'; did you mean 'N::X'?}}
#if __cplusplus <= 199711L
// expected-warning@-2 {{first declaration of class template specialization of 'X' outside namespace 'N' is a C++11 extension}}
#endif

namespace N {
  
template<> class X<char*> { /* ... */ };	// OK: X is a template
  
}
