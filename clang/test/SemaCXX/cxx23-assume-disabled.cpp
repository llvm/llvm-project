// RUN: %clang_cc1 -std=c++23 -x c++ %s -fno-assumptions -verify
// RUN: %clang_cc1 -std=c++23 -x c++ %s -fms-compatibility -verify
// RUN: %clang_cc1 -std=c++23 -x c++ %s -fno-assumptions -fexperimental-new-constant-interpreter -verify
// RUN: %clang_cc1 -std=c++23 -x c++ %s -fms-compatibility -fexperimental-new-constant-interpreter -verify

// expected-no-diagnostics

// We don't check assumptions at compile time if '-fno-assumptions' is passed,
// or if we're in MSVCCompat mode

constexpr bool f(bool x) {
  [[assume(x)]];
  return true;
}

static_assert(f(false));

