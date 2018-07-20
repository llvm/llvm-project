// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -verify %s -fchar8_t

#if defined(__cpp_char8_t) && __is_identifier(char8_t)
#error char8_t is an identifier under -fchar8_t
#endif

#if !defined(__cpp_char8_t) && !__is_identifier(char8_t)
#error char8_t is a keyword under -fno-char8_t
#endif

char8_t c8t;
#ifndef __cpp_char8_t
// expected-error@-2 {{unknown type}}
#else
// expected-no-diagnostics
#endif
