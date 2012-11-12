// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
__char16_t c16;
void f(__char32_t) { }
