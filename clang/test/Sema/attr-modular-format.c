//RUN: %clang_cc1 -fsyntax-only -verify %s

int printf(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float")));  // no-error
int myprintf(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float")));  // expected-error {{'modular_format' attribute requires 'format' attribute}}

int lprintf(const char *fmt, ...) __attribute__((modular_format(__modular_printf, L"__printf", L"float"), format(printf, 1, 2)));
// expected-warning@-1 2{{encoding prefix 'L' on an unevaluated string literal has no effect}}

int dupe(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float", "int", "float"), format(printf, 1, 2))); // expected-error {{duplicate aspect 'float' in 'modular_format' attribute}}
int multi_dupe(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float", "int", "float", "int"), format(printf, 1, 2))); // expected-error {{duplicate aspect 'float' in 'modular_format' attribute}} \
                                                                                                                                                                 // expected-error {{duplicate aspect 'int' in 'modular_format' attribute}}

// Test with multiple identical attributes on the same declaration.
int same_attr(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "float"), modular_format(__modular_printf, "__printf", "float"), format(printf, 1, 2))); // no-warning

// Test with multiple different attributes on the same declaration.
int diff_attr(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "float"), format(printf, 1, 2), modular_format(__modular_printf, "__printf", "int"))); // expected-error {{attribute 'modular_format' is already applied with different arguments}} expected-note {{conflicting attribute is here}}

int diff_attr2(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "float"), format(printf, 1, 2), modular_format(__modular_printf, "__other", "float"))); // expected-error {{attribute 'modular_format' is already applied with different arguments}} expected-note {{conflicting attribute is here}}

int diff_attr3(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "float"), format(printf, 1, 2), modular_format(__other, "__printf", "float"))); // expected-error {{attribute 'modular_format' is already applied with different arguments}} expected-note {{conflicting attribute is here}}

// Test with same attributes but different aspect order.
int diff_order(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "float", "int"), format(printf, 1, 2), modular_format(__modular_printf, "__printf", "int", "float"))); // no-error

// Test with multiple different attributes on a declaration and a redeclaration
int redecl(const char *fmt, ...) __attribute__((format(printf, 1, 2))); // no-error
int redecl(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "float"))); // expected-note {{conflicting attribute is here}}
int redecl(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "int"))); // expected-error {{attribute 'modular_format' is already applied with different arguments}}
