// RUN: %clang_cc1 -fsyntax-only -verify=cxx98 -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx11 -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx17 -std=c++17 %s

// The auto or register specifiers can be applied only to names of objects
// declared in a block (6.3) or to function parameters (8.4).

auto int ao;
// cxx11-error@-1 {{'auto' cannot be combined with a type specifier}}
// cxx17-error@-2 {{'auto' cannot be combined with a type specifier}}
// cxx98-error@-3 {{illegal storage class on file-scoped variable}}

auto void af();
// cxx11-error@-1 {{'auto' cannot be combined with a type specifier}}
// cxx17-error@-2 {{'auto' cannot be combined with a type specifier}}
// cxx98-error@-3 {{illegal storage class on function}}

register int ro;
// cxx98-error@-1 {{illegal storage class on file-scoped variable}}
// cxx11-error@-2 {{illegal storage class on file-scoped variable}}
// cxx11-warning@-3 {{'register' storage class specifier is deprecated and incompatible with C++17}}
// cxx17-error@-4 {{illegal storage class on file-scoped variable}}
// cxx17-error@-5 {{ISO C++17 does not allow 'register' storage class specifier}}

register void rf();
// cxx98-error@-1 {{illegal storage class on function}}
// cxx11-error@-2 {{illegal storage class on function}}
// cxx17-error@-3 {{illegal storage class on function}}

struct S {
  auto int ao;
  // cxx11-error@-1 {{'auto' cannot be combined with a type specifier}}
  // cxx17-error@-2 {{'auto' cannot be combined with a type specifier}}
  // cxx98-error@-3 {{storage class specified for a member declaration}}
  auto void af();
  // cxx11-error@-1 {{'auto' cannot be combined with a type specifier}}
  // cxx17-error@-2 {{'auto' cannot be combined with a type specifier}}
  // cxx98-error@-3 {{storage class specified for a member declaration}}

  register int ro;
  // cxx98-error@-1 {{storage class specified for a member declaration}}
  // cxx11-error@-2 {{storage class specified for a member declaration}}
  // cxx17-error@-3 {{storage class specified for a member declaration}}
  register void rf();
  // cxx98-error@-1 {{storage class specified for a member declaration}}
  // cxx11-error@-2 {{storage class specified for a member declaration}}
  // cxx17-error@-3 {{storage class specified for a member declaration}}
};

void foo(auto int ap, register int rp) {
  // cxx17-error@-1 {{'auto' cannot be combined with a type specifier}}
  // cxx17-error@-2 {{ISO C++17 does not allow 'register' storage class specifier}}
  // cxx11-error@-3 {{'auto' cannot be combined with a type specifier}}
  // cxx11-warning@-4 {{'register' storage class specifier is deprecated and incompatible with C++17}}
  auto int abo;
  // cxx11-error@-1 {{'auto' cannot be combined with a type specifier}}
  // cxx17-error@-2 {{'auto' cannot be combined with a type specifier}}
  auto void abf();
  // cxx11-error@-1 {{'auto' cannot be combined with a type specifier}}
  // cxx11-warning@-2 {{empty parentheses interpreted as a function declaration}}
  // cxx11-note@-3 {{replace parentheses with an initializer to declare a variable}}
  // cxx17-error@-4 {{'auto' cannot be combined with a type specifier}}
  // cxx17-warning@-5 {{empty parentheses interpreted as a function declaration}}
  // cxx17-note@-6 {{replace parentheses with an initializer to declare a variable}}
  // cxx98-error@-7 {{illegal storage class on function}}

  register int rbo;
  // cxx17-error@-1 {{ISO C++17 does not allow 'register' storage class specifier}}
  // cxx11-warning@-2 {{'register' storage class specifier is deprecated and incompatible with C++17}}

  register void rbf();
  // cxx98-error@-1 {{illegal storage class on function}}
  // cxx11-error@-2 {{illegal storage class on function}}
  // cxx17-error@-3 {{illegal storage class on function}}
}
