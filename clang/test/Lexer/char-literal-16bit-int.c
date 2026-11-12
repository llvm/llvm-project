// RUN: %clang_cc1 -triple avr-unknown-unknown -std=c11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple avr-unknown-unknown -x c++ -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// The value of a char32_t literal has all its bits, also where int has 16 of
// them.

#ifdef __cplusplus
#define _Static_assert static_assert
#endif

_Static_assert(sizeof(U'x') == 4, "");
_Static_assert(U'\U0001F600' == 0x1F600, "");
_Static_assert(U'\U0010FFFF' == 0x10FFFF, "");
_Static_assert(U'\U00010000' == 0x10000, "");
_Static_assert(u'￿' == 0xFFFF, "");
