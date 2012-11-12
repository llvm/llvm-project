// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace N {
  template<class T> class X; // expected-note {{'N::X' declared here}} \
                             // expected-note {{explicitly specialized declaration is here}}
}

// TODO: Don't add a namespace qualifier to the template if it would trigger
// the warning about the specialization being outside of the namespace.
template<> class X<int> { /* ... */ };	// expected-error {{no template named 'X'; did you mean 'N::X'?}} \
                                        // expected-warning {{first declaration of class template specialization of 'X' outside namespace 'N' is a C++11 extension}}

namespace N {
  
template<> class X<char*> { /* ... */ };	// OK: X is a template
  
}
