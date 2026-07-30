// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=cxx,expected %s

template <class a> using __impl_of = a; // expected-note {{'__impl_of' declared here}} \
                                           expected-note {{template is declared here}}
struct {                                // expected-error {{anonymous structs and classes must be class members}} \
                                           expected-note {{to match this '{'}}
  __impl_;                              // expected-error {{no template named '__impl_'; did you mean '__impl_of'?}} \
                                           expected-error {{cannot specify deduction guide for alias template '__impl_of'}} \
                                           expected-error {{expected ';' after struct}}
                                        // expected-error {{expected '}'}}
