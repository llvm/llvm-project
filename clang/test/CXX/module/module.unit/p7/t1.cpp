// RUN: %clang_cc1 -std=c++20 %s -verify
// expected-no-diagnostics
module;

#include "Inputs/h1.h"

export module x;

extern "C" void foo() {
  return;
}

extern "C" {
void bar() {
  return;
}
int baz() {
  return 3;
}
double double_func() {
  return 5.0;
}
}

extern "C++" {
void bar_cpp() {
  return;
}
int baz_cpp() {
  return 3;
}
double double_func_cpp() {
  return 5.0;
}
}
