// RUN: %clang_cc1 -x c -std=c11 -Werror %s
// RUN: %clang_cc1 -x c++ -std=c++11 -Werror %s
// RUN: %clang_cc1 -x c -std=c11 -Wc99-compat -verify %s
// RUN: %clang_cc1 -x c++ -std=c++11 -Wc++98-compat -verify %s

#ifndef __cplusplus
typedef __CHAR16_TYPE__ char16_t;
typedef __CHAR32_TYPE__ char32_t;
#else
// expected-warning@17 {{'char16_t' type specifier is incompatible with C++98}}
// expected-warning@18 {{'char32_t' type specifier is incompatible with C++98}}
// expected-warning@20 {{'char16_t' type specifier is incompatible with C++98}}
// expected-warning@21 {{'char32_t' type specifier is incompatible with C++98}}
#endif

const char *a = u8"abcd"; // expected-warning {{unicode literals are incompatible with}}
const char16_t *b = u"abcd"; // expected-warning {{unicode literals are incompatible with}}
const char32_t *c = U"abcd"; // expected-warning {{unicode literals are incompatible with}}

char16_t d = u'a'; // expected-warning {{unicode literals are incompatible with}}
char32_t e = U'a'; // expected-warning {{unicode literals are incompatible with}}
