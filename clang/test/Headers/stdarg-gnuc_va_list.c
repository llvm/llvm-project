// RUN: %clang_cc1 -fsyntax-only -verify -Wsystem-headers -std=c99 %s
// expected-no-diagnostics

// Check that no warnings are emitted from stdarg.h if __gnuc_va_list has
// previously been defined in another header file.
typedef __builtin_va_list __va_list;
typedef __va_list __gnuc_va_list;
#define __GNUC_VA_LIST

#include <stdarg.h>
