// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -fsyntax-only -verify %s

// expected-no-diagnostics
typedef __float128 f128_t;
