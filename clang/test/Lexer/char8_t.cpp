// RUN: %clang_cc1 -std=c++20 -verify %s -DCHAR8_T
// RUN: %clang_cc1 -std=c++20 -verify %s -fchar8_t -DCHAR8_T
// RUN: %clang_cc1 -std=c++17 -verify %s -fchar8_t -DCHAR8_T

// RUN: %clang_cc1 -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s -fno-char8_t
// RUN: %clang_cc1 -std=c++20 -verify %s -fno-char8_t

#if defined(__cpp_char8_t) != defined(CHAR8_T)
#error wrong setting for __cpp_char_t
#endif

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
