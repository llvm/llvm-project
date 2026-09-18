// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fsyntax-only -verify %s
// expected-no-diagnostics

// SPIR-V's OpenCL metadata must not impose OpenCL initialization restrictions.
__attribute__((device)) void atomic_init() {
  _Atomic(int) x = 0;
}
