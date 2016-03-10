// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace N1 {

namespace __attribute__((__abi_tag__)) {}
// expected-warning@-1 {{'abi_tag' attribute on non-inline namespace ignored}}

namespace N __attribute__((__abi_tag__)) {}
// expected-warning@-1 {{'abi_tag' attribute on non-inline namespace ignored}}

} // namespace N1

namespace N2 {

inline namespace __attribute__((__abi_tag__)) {}
// expected-warning@-1 {{'abi_tag' attribute on anonymous namespace ignored}}

inline namespace N __attribute__((__abi_tag__)) {}
// FIXME: remove this warning as soon as attribute fully supported.
// expected-warning@-2 {{'__abi_tag__' attribute ignored}}

} // namespcace N2

__attribute__((abi_tag("B", "A"))) extern int a1;
// FIXME: remove this warning as soon as attribute fully supported.
// expected-warning@-2 {{'abi_tag' attribute ignored}}

__attribute__((abi_tag("A", "B"))) extern int a1;
// expected-note@-1 {{previous declaration is here}}
// FIXME: remove this warning as soon as attribute fully supported.
// expected-warning@-3 {{'abi_tag' attribute ignored}}

__attribute__((abi_tag("A", "C"))) extern int a1;
// expected-error@-1 {{'abi_tag' C missing in original declaration}}
// FIXME: remove this warning as soon as attribute fully supported.
// expected-warning@-3 {{'abi_tag' attribute ignored}}

extern int a2;
// expected-note@-1 {{previous declaration is here}}
__attribute__((abi_tag("A")))extern int a2;
// expected-error@-1 {{cannot add 'abi_tag' attribute in a redeclaration}}
// FIXME: remove this warning as soon as attribute fully supported.
// expected-warning@-3 {{'abi_tag' attribute ignored}}
