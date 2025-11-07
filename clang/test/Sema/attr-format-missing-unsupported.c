// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ -std=c++98 -Wmissing-format-attribute %s

typedef unsigned long size_t;
typedef long ssize_t;
typedef __builtin_va_list va_list;

__attribute__((format(printf, 1, 0)))
int vprintf(const char *, va_list);

// Test that diagnostic is disabled when the standard doesn't support a portable attribute syntax.
// expected-no-diagnostics

void f1(char *out, va_list args) // #f1
{
  vprintf(out, args);
}
