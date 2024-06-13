// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2 -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -verify=overload -verify-ignore-unexpected=error,note -emit-llvm -o - %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

void test_bfloat(const bfloat16_t *const_bf16_ptr, svbfloat16_t bf16, svbfloat16x2_t bf16x2)
{
  // expected-error@+2 {{'svwhilerw_bf16' needs target feature sve2,bf16}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2,bf16}}
  SVE_ACLE_FUNC(svwhilerw,_bf16,,)(const_bf16_ptr, const_bf16_ptr);
  // expected-error@+2 {{'svtbx_bf16' needs target feature sve2,bf16}}
  // overload-error@+1 {{'svtbx' needs target feature sve2,bf16}}
  SVE_ACLE_FUNC(svtbx,_bf16,,)(bf16, bf16, svundef_u16());
  // expected-error@+2 {{'svtbl2_bf16' needs target feature sve2,bf16}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2,bf16}}
  SVE_ACLE_FUNC(svtbl2,_bf16,,)(bf16x2, svundef_u16());
  // expected-error@+2 {{'svwhilewr_bf16' needs target feature sve2,bf16}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2,bf16}}
  SVE_ACLE_FUNC(svwhilewr,_bf16,,)(const_bf16_ptr, const_bf16_ptr);
}
