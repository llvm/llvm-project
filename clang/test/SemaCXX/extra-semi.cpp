// RUN: %clang_cc1 -verify -std=c++98 -Wextra-semi %s
// RUN: %clang_cc1 -verify -std=c++03 -Wextra-semi %s
// RUN: %clang_cc1 -verify -std=c++11 -Wextra-semi %s
// RUN: %clang_cc1 -verify -std=c++17 -Wextra-semi %s
// RUN: %clang_cc1 -verify -std=c++2a -Wextra-semi %s
// RUN: %clang_cc1 -verify -Weverything -Wno-c++98-compat %s
// RUN: %clang_cc1 -verify -Weverything -Wno-c++98-compat-pedantic -Wc++98-compat-extra-semi %s

// Last RUN line checks that c++98-compat-extra-semi can still be re-enabled.

void F();

void F(){}
;
#if __cplusplus < 201103L
// expected-warning@-2{{extra ';' outside of a function is a C++11 extension}}
#else
// expected-warning@-4{{extra ';' outside of a function is incompatible with C++98}}
#endif

namespace ns {
class C {
  void F() const;
};
}
; // expected-warning {{extra ';' outside of a function is}}

void ns::C::F() const {}
; // expected-warning {{extra ';' outside of a function is}}
