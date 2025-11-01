// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// The auto or register specifiers can be applied only to names of objects
// declared in a block (6.3) or to function parameters (8.4).

auto int ao;
#if __cplusplus >= 201103L // C++11 or later
// expected-error@-2 {{'auto' cannot be combined with a type specifier in C++}}
#else
// expected-error@-4 {{illegal storage class on file-scoped variable}}
#endif

auto void af();
#if __cplusplus >= 201103L // C++11 or later
// expected-error@-2 {{'auto' cannot be combined with a type specifier in C++}}
#else
// expected-error@-4 {{illegal storage class on function}}
#endif

register int ro; // expected-error {{illegal storage class on file-scoped variable}}
#if __cplusplus >= 201703L
// expected-error@-2 {{ISO C++17 does not allow 'register' storage class specifier}}
#elif __cplusplus >= 201103L
// expected-warning@-4 {{'register' storage class specifier is deprecated}}
#endif

register void rf(); // expected-error {{illegal storage class on function}}

struct S {
  auto int ao;
#if __cplusplus >= 201103L // C++11 or later
// expected-error@-2 {{'auto' cannot be combined with a type specifier in C++}}
#else
// expected-error@-4 {{storage class specified for a member declaration}}
#endif
  auto void af();
#if __cplusplus >= 201103L // C++11 or later
// expected-error@-2 {{'auto' cannot be combined with a type specifier in C++}}
#else
// expected-error@-4 {{storage class specified for a member declaration}}
#endif

  register int ro; // expected-error {{storage class specified for a member declaration}}
  register void rf(); // expected-error {{storage class specified for a member declaration}}
};

void foo(auto int ap, register int rp) {
#if __cplusplus >= 201703L
// expected-error@-2 {{'auto' cannot be combined with a type specifier in C++}}
// expected-error@-3 {{ISO C++17 does not allow 'register' storage class specifier}}
#elif __cplusplus >= 201103L
// expected-error@-5 {{'auto' cannot be combined with a type specifier in C++}}
// expected-warning@-6 {{'register' storage class specifier is deprecated}}
#endif
  auto int abo;
#if __cplusplus >= 201103L // C++11 or later
// expected-error@-2 {{'auto' cannot be combined with a type specifier in C++}}
#endif
  auto void abf();
#if __cplusplus >= 201103L // C++11 or later
// expected-error@-2 {{'auto' cannot be combined with a type specifier in C++}}
#else
// expected-error@-4 {{illegal storage class on function}}
#endif

  register int rbo;
#if __cplusplus >= 201703L
// expected-error@-2 {{ISO C++17 does not allow 'register' storage class specifier}}
#elif __cplusplus >= 201103L
// expected-warning@-4 {{'register' storage class specifier is deprecated}}
#endif

  register void rbf(); // expected-error {{illegal storage class on function}}
}
