// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++0x-compat %s

// The auto or register specifiers can be applied only to names of objects
// declared in a block (6.3) or to function parameters (8.4).

auto int ao; // expected-error {{illegal storage class on file-scoped variable}}
auto void af(); // expected-error {{illegal storage class on function}}

register int ro; // expected-error {{illegal storage class on file-scoped variable}}
register void rf(); // expected-error {{illegal storage class on function}}

struct S {
  auto int ao; // expected-error {{storage class specified for a member declaration}}
  auto void af(); // expected-error {{storage class specified for a member declaration}}

  register int ro; // expected-error {{storage class specified for a member declaration}}
  register void rf(); // expected-error {{storage class specified for a member declaration}}
};

void foo(auto int ap, register int rp) {
  auto int abo;
  auto void abf(); // expected-error {{illegal storage class on function}}

  register int rbo;
  register void rbf(); // expected-error {{illegal storage class on function}}
}
