// RUN: %clang_cc1 %s -emit-llvm -o - -std=c++11 -verify
// expected-no-diagnostics

static_assert(true, "");

void f() {
  static_assert(true, "");
}
