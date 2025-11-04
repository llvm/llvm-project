//RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stdarg.h>

int printf(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float")));  // no-error
int myprintf(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float")));  // expected-error {{'modular_format' attribute requires 'format' attribute}}

