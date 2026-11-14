// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.0-library %s -fhlsl-strict-availability \
// RUN:   -fsyntax-only -verify
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -fhlsl-strict-availability \
// RUN:   -fsyntax-only -verify
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fhlsl-strict-availability \
// RUN:   -fsyntax-only -verify

// expected-no-diagnostics

bool test_availability(bool cond) {
  return QuadAll(cond);
}
