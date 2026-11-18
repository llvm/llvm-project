// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -fhlsl-strict-availability \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fhlsl-strict-availability \
// RUN:   -fsyntax-only -verify=sm67

// sm67-no-diagnostics

bool test_availability(bool cond) {
  // expected-error@+1 {{'QuadAll' is only available on Shader Model 6.7}}
  return QuadAll(cond);
}
