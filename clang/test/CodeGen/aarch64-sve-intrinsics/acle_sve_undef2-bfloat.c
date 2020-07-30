// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

#include <arm_sve.h>

svbfloat16x2_t test_svundef2_bf16()
{
  // CHECK-LABEL: test_svundef2_bf16
  // CHECK: ret <vscale x 16 x bfloat> undef
  // expected-warning@+1 {{implicit declaration of function 'svundef2_bf16'}}
  return svundef2_bf16();
}
