// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

extern "C" int printf(const char *fmt, ...)
    __attribute__((modular_format(__modular_printf, "__printf", "float")));
