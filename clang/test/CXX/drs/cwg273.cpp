// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors

// expected-no-diagnostics

#include <stdarg.h>
#include <stddef.h>
namespace cwg273 { // cwg273: 2.7
  struct A {
    int n;
  };
  void operator&(A);
  void f(A a, ...) {
    offsetof(A, n);
    va_list val;
    va_start(val, a);
    va_end(val);
  }
} // namespace cwg273
