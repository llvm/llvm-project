// RUN: %clang_cc1 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

#include <arm_sve.h>

svbfloat16_t test_svundef_bf16()
{
  // CHECK-LABEL: test_svundef_bf16
  // CHECK: ret <vscale x 8 x bfloat> undef
  // expected-warning@+1 {{implicit declaration of function 'svundef_bf16'}}
  return svundef_bf16();
}
