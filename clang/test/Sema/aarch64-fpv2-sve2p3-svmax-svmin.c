// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2p3 -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -verify=overload -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2p3 -verify=overload -verify-ignore-unexpected=error,note -emit-llvm -o - %s
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3_UNUSED,A4) A1##A4
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test(svbool_t pg, float16_t f16, float32_t f32, float64_t f64) {
  // expected-error@+2 {{'svmaxmg_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f16, _m)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxmg_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f16, _x)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxmg_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f16, _z)(pg, svundef_f16(), svundef_f16());

  // expected-error@+2 {{'svmaxmg_n_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f16, _m)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svmaxmg_n_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f16, _x)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svmaxmg_n_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f16, _z)(pg, svundef_f16(), f16);

  // expected-error@+2 {{'svmaxmg_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f32, _m)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxmg_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f32, _x)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxmg_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f32, _z)(pg, svundef_f32(), svundef_f32());

  // expected-error@+2 {{'svmaxmg_n_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f32, _m)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svmaxmg_n_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f32, _x)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svmaxmg_n_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f32, _z)(pg, svundef_f32(), f32);

  // expected-error@+2 {{'svmaxmg_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f64, _m)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxmg_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f64, _x)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxmg_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,,_f64, _z)(pg, svundef_f64(), svundef_f64());

  // expected-error@+2 {{'svmaxmg_n_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f64, _m)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svmaxmg_n_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f64, _x)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svmaxmg_n_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmg,_n,_f64, _z)(pg, svundef_f64(), f64);

  // expected-error@+2 {{'svminmg_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f16, _m)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svminmg_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f16, _x)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svminmg_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f16, _z)(pg, svundef_f16(), svundef_f16());

  // expected-error@+2 {{'svminmg_n_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f16, _m)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svminmg_n_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f16, _x)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svminmg_n_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f16, _z)(pg, svundef_f16(), f16);

  // expected-error@+2 {{'svminmg_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f32, _m)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svminmg_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f32, _x)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svminmg_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f32, _z)(pg, svundef_f32(), svundef_f32());

  // expected-error@+2 {{'svminmg_n_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f32, _m)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svminmg_n_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f32, _x)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svminmg_n_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f32, _z)(pg, svundef_f32(), f32);

  // expected-error@+2 {{'svminmg_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f64, _m)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svminmg_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f64, _x)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svminmg_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,,_f64, _z)(pg, svundef_f64(), svundef_f64());

  // expected-error@+2 {{'svminmg_n_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f64, _m)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svminmg_n_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f64, _x)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svminmg_n_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmg_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmg,_n,_f64, _z)(pg, svundef_f64(), f64);

  // expected-error@+2 {{'svmaxinm_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f16, _m)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxinm_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f16, _x)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxinm_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f16, _z)(pg, svundef_f16(), svundef_f16());

  // expected-error@+2 {{'svmaxinm_n_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f16, _m)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svmaxinm_n_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f16, _x)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svmaxinm_n_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f16, _z)(pg, svundef_f16(), f16);

  // expected-error@+2 {{'svmaxinm_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f32, _m)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxinm_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f32, _x)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxinm_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f32, _z)(pg, svundef_f32(), svundef_f32());

  // expected-error@+2 {{'svmaxinm_n_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f32, _m)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svmaxinm_n_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f32, _x)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svmaxinm_n_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f32, _z)(pg, svundef_f32(), f32);

  // expected-error@+2 {{'svmaxinm_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f64, _m)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxinm_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f64, _x)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxinm_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,,_f64, _z)(pg, svundef_f64(), svundef_f64());

  // expected-error@+2 {{'svmaxinm_n_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f64, _m)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svmaxinm_n_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f64, _x)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svmaxinm_n_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxinm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxinm,_n,_f64, _z)(pg, svundef_f64(), f64);

  // expected-error@+2 {{'svmininm_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f16, _m)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmininm_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f16, _x)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmininm_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f16, _z)(pg, svundef_f16(), svundef_f16());

  // expected-error@+2 {{'svmininm_n_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f16, _m)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svmininm_n_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f16, _x)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svmininm_n_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f16, _z)(pg, svundef_f16(), f16);

  // expected-error@+2 {{'svmininm_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f32, _m)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmininm_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f32, _x)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmininm_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f32, _z)(pg, svundef_f32(), svundef_f32());

  // expected-error@+2 {{'svmininm_n_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f32, _m)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svmininm_n_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f32, _x)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svmininm_n_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f32, _z)(pg, svundef_f32(), f32);

  // expected-error@+2 {{'svmininm_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f64, _m)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmininm_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f64, _x)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmininm_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,,_f64, _z)(pg, svundef_f64(), svundef_f64());

  // expected-error@+2 {{'svmininm_n_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f64, _m)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svmininm_n_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f64, _x)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svmininm_n_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmininm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmininm,_n,_f64, _z)(pg, svundef_f64(), f64);

  // expected-error@+2 {{'svmaxmginm_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f16, _m)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxmginm_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f16, _x)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxmginm_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f16, _z)(pg, svundef_f16(), svundef_f16());

  // expected-error@+2 {{'svmaxmginm_n_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f16, _m)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svmaxmginm_n_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f16, _x)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svmaxmginm_n_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f16, _z)(pg, svundef_f16(), f16);

  // expected-error@+2 {{'svmaxmginm_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f32, _m)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxmginm_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f32, _x)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxmginm_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f32, _z)(pg, svundef_f32(), svundef_f32());

  // expected-error@+2 {{'svmaxmginm_n_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f32, _m)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svmaxmginm_n_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f32, _x)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svmaxmginm_n_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f32, _z)(pg, svundef_f32(), f32);

  // expected-error@+2 {{'svmaxmginm_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f64, _m)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxmginm_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f64, _x)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxmginm_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,,_f64, _z)(pg, svundef_f64(), svundef_f64());

  // expected-error@+2 {{'svmaxmginm_n_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f64, _m)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svmaxmginm_n_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f64, _x)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svmaxmginm_n_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svmaxmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svmaxmginm,_n,_f64, _z)(pg, svundef_f64(), f64);

  // expected-error@+2 {{'svminmginm_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f16, _m)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svminmginm_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f16, _x)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svminmginm_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f16, _z)(pg, svundef_f16(), svundef_f16());

  // expected-error@+2 {{'svminmginm_n_f16_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f16, _m)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svminmginm_n_f16_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f16, _x)(pg, svundef_f16(), f16);
  // expected-error@+2 {{'svminmginm_n_f16_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f16, _z)(pg, svundef_f16(), f16);

  // expected-error@+2 {{'svminmginm_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f32, _m)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svminmginm_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f32, _x)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svminmginm_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f32, _z)(pg, svundef_f32(), svundef_f32());

  // expected-error@+2 {{'svminmginm_n_f32_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f32, _m)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svminmginm_n_f32_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f32, _x)(pg, svundef_f32(), f32);
  // expected-error@+2 {{'svminmginm_n_f32_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f32, _z)(pg, svundef_f32(), f32);

  // expected-error@+2 {{'svminmginm_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f64, _m)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svminmginm_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f64, _x)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svminmginm_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,,_f64, _z)(pg, svundef_f64(), svundef_f64());

  // expected-error@+2 {{'svminmginm_n_f64_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_m' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f64, _m)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svminmginm_n_f64_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_x' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f64, _x)(pg, svundef_f64(), f64);
  // expected-error@+2 {{'svminmginm_n_f64_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  // overload-error@+1 {{'svminmginm_z' needs target feature (sve,(sve2p3|sme2p3),fpv2)|(sme,(sve2p3|sme2p3),fpv2)}}
  SVE_ACLE_FUNC(svminmginm,_n,_f64, _z)(pg, svundef_f64(), f64);
}
