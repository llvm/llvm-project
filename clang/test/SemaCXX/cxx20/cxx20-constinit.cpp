// RUN: %clang_cc1 %s -std=c++20 -verify
// expected-no-diagnostics

constinit int a __attribute__((weak)) = 0;
