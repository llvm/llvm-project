// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn-amd-amdhsa -fcuda-is-device -fsyntax-only -verify %s

// expected-no-diagnostics

__fp16 *testFP16AsArgAndReturn(__fp16 *x) {
  return x;
}
