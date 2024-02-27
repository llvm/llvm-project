// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both -Wno-unused-value %s
// RUN: %clang_cc1 -verify=ref,both -Wno-unused-value %s

// expected-no-diagnostics
// ref-no-diagnostics

void blah() {
  __complex__ unsigned xx;
  __complex__ signed yy;
  __complex__ int result;

  /// The following line calls into the constant interpreter.
  result = xx * yy;
}
