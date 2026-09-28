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

} // namespace N2

namespace N3 {
inline namespace AbsentOld {}
inline namespace AbsentOld __attribute__((__abi_tag__)) {}
// expected-warning@-2 {{no 'abi_tag' prevents applying 'abi_tag' AbsentOld later}}
// expected-note@-2 {{declared here}}

inline namespace AbsentNew __attribute__((__abi_tag__)) {}
inline namespace AbsentNew {}
// No tags on a namespace reopening can be deliberate, no diagnostic.

inline namespace Different __attribute__((abi_tag("A"))) {}
inline namespace Different __attribute__((abi_tag("B"))) {}
// expected-warning@-2 {{'abi_tag' A prevents applying 'abi_tag' B later}}
// expected-note@-2 {{declared here}}
inline namespace Different __attribute__((abi_tag("A"))) {}
// No error as we compare with the canonical namespace decl, not with the previous one.

inline namespace MultipleTags __attribute__((abi_tag("A", "B"))) {}
inline namespace MultipleTags __attribute__((abi_tag("X", "Y", "B"))) {}
// expected-warning@-2 {{'abi_tag' A, B prevents applying 'abi_tag' B, X, Y later}}
// expected-note@-2 {{declared here}}
} // namespace N3

__attribute__((abi_tag("B", "A"))) extern int a1;

__attribute__((abi_tag("A", "B"))) extern int a1;
// expected-note@-1 {{previous declaration is here}}

__attribute__((abi_tag("A", "C"))) extern int a1;
// expected-error@-1 {{'abi_tag' C missing in original declaration}}

extern int a2;
// expected-note@-1 {{previous declaration is here}}
__attribute__((abi_tag("A")))extern int a2;
// expected-error@-1 {{cannot add 'abi_tag' attribute in a redeclaration}}
