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
// expected-error@-1 {{'abi_tag' AbsentOld is ignored, applying no 'abi_tag'}}
// expected-note@-3 {{previous declaration is here}}

inline namespace AbsentNew __attribute__((__abi_tag__)) {}
inline namespace AbsentNew {}
// expected-error@-1 {{absent 'abi_tag' attribute is ignored, applying 'abi_tag' AbsentNew}}
// expected-note@-3 {{previous declaration is here}}

inline namespace Different __attribute__((abi_tag("A"))) {}
inline namespace Different __attribute__((abi_tag("B"))) {}
// expected-error@-1 {{'abi_tag' B is ignored, applying 'abi_tag' A}}
// expected-note@-3 {{previous declaration is here}}
inline namespace Different __attribute__((abi_tag("A"))) {}
// No error as we compare with the canonical namespace decl, not with the previous one.

inline namespace MultipleTags __attribute__((abi_tag("A", "B"))) {}
inline namespace MultipleTags __attribute__((abi_tag("X", "Y", "B"))) {}
// expected-error@-1 {{'abi_tag' B, X, Y is ignored, applying 'abi_tag' A, B}}
// expected-note@-3 {{previous declaration is here}}
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
