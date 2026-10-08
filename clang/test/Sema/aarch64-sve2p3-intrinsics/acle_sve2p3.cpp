// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -verify=overload -verify-ignore-unexpected=error,note -emit-llvm -o - %s
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3_UNUSED) A1
#else
#define SVE_ACLE_FUNC(A1,A2,A3) A1##A2##A3
#endif

void test_svabal(int8_t s8, int16_t s16, int32_t s32, uint8_t u8, uint16_t u16, uint32_t u32)
{
  // expected-error@+2 {{'svabal_s64' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,,_s64)(svundef_s64(), svundef_s32(), svundef_s32());

  // expected-error@+2 {{'svabal_n_s64' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,_n,_s64)(svundef_s64(), svundef_s32(), s32);

  // expected-error@+2 {{'svabal_s32' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,,_s32)(svundef_s32(), svundef_s16(), svundef_s16());

  // expected-error@+2 {{'svabal_n_s32' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,_n,_s32)(svundef_s32(), svundef_s16(), s16);

  // expected-error@+2 {{'svabal_s16' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,,_s16)(svundef_s16(), svundef_s8(), svundef_s8());

  // expected-error@+2 {{'svabal_n_s16' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,_n,_s16)(svundef_s16(), svundef_s8(), s8);

  // expected-error@+2 {{'svabal_u64' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,,_u64)(svundef_u64(), svundef_u32(), svundef_u32());

  // expected-error@+2 {{'svabal_n_u64' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,_n,_u64)(svundef_u64(), svundef_u32(), u32);

  // expected-error@+2 {{'svabal_u32' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,,_u32)(svundef_u32(), svundef_u16(), svundef_u16());

  // expected-error@+2 {{'svabal_n_u32' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,_n,_u32)(svundef_u32(), svundef_u16(), u16);

  // expected-error@+2 {{'svabal_u16' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,,_u16)(svundef_u16(), svundef_u8(), svundef_u8());

  // expected-error@+2 {{'svabal_n_u16' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  // overload-error@+1 {{'svabal' needs target feature (sve,(sve2p3|sme2p3))|(sme,(sve2p3|sme2p3))}}
  SVE_ACLE_FUNC(svabal,_n,_u16)(svundef_u16(), svundef_u8(), u8);
}
