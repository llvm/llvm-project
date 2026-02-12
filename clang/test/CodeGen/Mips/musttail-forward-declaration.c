// RUN: %clang_cc1 %s -triple mipsel-unknown-linux-gnu -o /dev/null -emit-llvm -verify

// Test that a forward declaration that is later defined in the same TU
// is allowed for musttail calls.

int func(int i);

int caller(int i) {
  // expected-no-diagnostics
  [[clang::musttail]] return func(i);
}

int func(int i) {
  return i + 1;
}
