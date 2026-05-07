// RUN: %clang_cc1 -x c++ -fsyntax-only -verify %s

extern "C" int printf(const char *fmt, ...)
    __attribute__((modular_format(__modular_printf, "__printf", "float")));
extern "C" int asprintf(char **buf, const char *fmt, ...)
    __attribute__((modular_format(__asprintf_modular, "__printf", "float")));
extern "C" int vasprintf(char **buf, const char *fmt, __builtin_va_list ap)
    __attribute__((modular_format(__vasprintf_modular, "__printf", "float")));

int myprintf(const char *fmt, ...) __attribute__((
    modular_format(__modular_printf, "__printf", "float")));
// expected-error@-2 {{'modular_format' attribute requires 'format' attribute}}

namespace ns {
int asprintf(char **buf, const char *fmt, ...)
    __attribute__((modular_format(__asprintf_modular, "__printf", "float")));
// expected-error@-2 {{'modular_format' attribute requires 'format' attribute}}
} // namespace ns
