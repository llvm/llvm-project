// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// The auto or register specifiers can be applied only to names of objects
// declared in a block (6.3) or to function parameters (8.4).

auto int ao; // expected-error {{illegal storage class on file-scoped variable}}
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
#endif

auto void af(); // expected-error {{illegal storage class on function}}
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
#endif

register int ro; // expected-error {{illegal storage class on file-scoped variable}}
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'register' storage class specifier is deprecated}}
#endif

register void rf(); // expected-error {{illegal storage class on function}}

struct S {
  auto int ao; // expected-error {{storage class specified for a member declaration}}
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
#endif
  auto void af(); // expected-error {{storage class specified for a member declaration}}
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
#endif

  register int ro; // expected-error {{storage class specified for a member declaration}}
  register void rf(); // expected-error {{storage class specified for a member declaration}}
};

void foo(auto int ap, register int rp) {
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
#endif
  auto int abo;
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
#endif
  auto void abf(); // expected-error {{illegal storage class on function}}
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
#endif

  register int rbo;
#if __cplusplus >= 201103L // C++11 or later
// expected-warning@-2 {{'register' storage class specifier is deprecated}}
#endif

  register void rbf(); // expected-error {{illegal storage class on function}}
}
