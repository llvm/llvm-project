// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace foo {}
template <class type> foo::typel::x f();
// expected-error@-1 {{no member named 'typel' in namespace 'foo'}}
