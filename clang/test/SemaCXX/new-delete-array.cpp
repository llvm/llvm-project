// RUN: %clang_cc1 -fsyntax-only -std=c++17 %s -verify=cxx17
// RUN: %clang_cc1 -fsyntax-only -std=c++14 %s -verify=cxx14

namespace PR51547 {
template<class> struct A; // cxx14-note {{template is declared here}}
auto p = new A[]{}; // cxx14-error {{use of class template 'A' requires template arguments}} \
                       cxx17-error {{cannot form array of deduced class template specialization type}}
}

