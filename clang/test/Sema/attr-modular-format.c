//RUN: %clang_cc1 -fsyntax-only -verify %s

int printf(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float")));  // no-error
int myprintf(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float")));  // expected-error {{'modular_format' attribute requires 'format' attribute}}

int dupe(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float", "int", "float"), format(printf, 1, 2))); // expected-error {{duplicate aspect 'float' in 'modular_format' attribute}}
int multi_dupe(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float", "int", "float", "int"), format(printf, 1, 2))); // expected-error {{duplicate aspect 'float' in 'modular_format' attribute}} \
                                                                                                                                                                 // expected-error {{duplicate aspect 'int' in 'modular_format' attribute}}

