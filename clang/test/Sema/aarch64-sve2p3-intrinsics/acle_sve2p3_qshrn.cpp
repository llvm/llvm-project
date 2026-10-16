// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -verify=overload -verify-ignore-unexpected=error,note -emit-llvm -o - %s
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test(svint32x2_t s32x2, svint16x2_t s16x2, svint8x2_t s8x2, svuint32x2_t u32x2, svuint16x2_t u16x2, svuint8x2_t u8x2)
{
  // expected-error@+2 {{'svqshrn_n_s8_s16_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqshrn_s8' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqshrn,_n,_s8,_s16_x2)(s16x2, 8);

  // expected-error@+2 {{'svqshrn_n_s16_s32_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqshrn_s16' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqshrn,_n,_s16,_s32_x2)(s32x2, 16);

  // expected-error@+2 {{'svqshrn_n_u8_u16_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqshrn_u8' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqshrn,_n,_u8,_u16_x2)(u16x2, 8);

  // expected-error@+2 {{'svqshrn_n_u16_u32_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqshrn_u16' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqshrn,_n,_u16,_u32_x2)(u32x2, 16);

  // expected-error@+2 {{'svqshrun_n_u16_s32_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqshrun_u16' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqshrun,_n,_u16,_s32_x2)(s32x2, 16);

  // expected-error@+2 {{'svqshrun_n_u8_s16_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqshrun_u8' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqshrun,_n,_u8,_s16_x2)(s16x2, 8);

  // expected-error@+2 {{'svqrshrn_n_s8_s16_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqrshrn_s8' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqrshrn,_n,_s8,_s16_x2)(s16x2, 8);

  // expected-error@+2 {{'svqrshrn_n_u8_u16_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqrshrn_u8' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqrshrn,_n,_u8,_u16_x2)(u16x2, 8);

  // expected-error@+2 {{'svqrshrun_n_u8_s16_x2' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svqrshrun_u8' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svqrshrun,_n,_u8,_s16_x2)(s16x2, 8);
}
