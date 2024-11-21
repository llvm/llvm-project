// RUN: %clang_cc1 -o - -triple amdgcn-amd-amdhsa -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -o - -triple spirv64-amd-amdhsa -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -o - -triple x86_64-unknown-gnu-linux -fsyntax-only -verify -xhip %s

// expected-no-diagnostics

__fp16 testFP16AsArgAndReturn(__fp16 x) {
  return x;
}
