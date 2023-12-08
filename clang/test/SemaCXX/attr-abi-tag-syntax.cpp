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

} // namespcace N2

__attribute__((abi_tag("B", "A"))) extern int a1;

__attribute__((abi_tag("A", "B"))) extern int a1;
// expected-note@-1 {{previous declaration is here}}

__attribute__((abi_tag("A", "C"))) extern int a1;
// expected-error@-1 {{'abi_tag' C missing in original declaration}}

extern int a2;
// expected-note@-1 {{previous declaration is here}}
__attribute__((abi_tag("A")))extern int a2;
// expected-error@-1 {{cannot add 'abi_tag' attribute in a redeclaration}}
