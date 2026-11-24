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

// TODO: Test vulkan targets after 
// https://github.com/llvm/llvm-project/issues/243266 is resolved.

bool test_availability(bool cond) {
  return QuadAll(cond);
}
