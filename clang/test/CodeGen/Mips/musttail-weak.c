// RUN: %clang_cc1 %s -triple mipsel-unknown-linux-gnu -o /dev/null -emit-llvm -verify

// Test musttail with weak functions.
// Weak functions can be interposed, so they are not considered DSO-local.

__attribute__((weak)) int weak_func(int i) {
  return i;
}

int caller(int i) {
  // expected-error@+1 {{'musttail' attribute for this call is impossible because calls outside the current linkage unit cannot be tail called on MIPS}}
  [[clang::musttail]] return weak_func(i);
}
