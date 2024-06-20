// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -verify=overload -verify-ignore-unexpected=error,note -emit-llvm -o - %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

int8_t i8;
int16_t i16;
int32_t i32;
uint8_t u8;
uint16_t u16;
uint32_t u32;
uint64_t u64;
int64_t i64;
int64_t *i64_ptr;
uint64_t *u64_ptr;
float64_t *f64_ptr;
int32_t *i32_ptr;
uint32_t *u32_ptr;
float32_t *f32_ptr;
int16_t *i16_ptr;
uint16_t *u16_ptr;
int8_t *i8_ptr;
uint8_t *u8_ptr;

void test(svbool_t pg, const int8_t *const_i8_ptr, const uint8_t *const_u8_ptr,
          const int16_t *const_i16_ptr, const uint16_t *const_u16_ptr,
          const int32_t *const_i32_ptr, const uint32_t *const_u32_ptr,
          const int64_t *const_i64_ptr, const uint64_t *const_u64_ptr,
          const float16_t *const_f16_ptr, const float32_t *const_f32_ptr, const float64_t *const_f64_ptr)
{
  // expected-error@+2 {{'svhistseg_s8' needs target feature sve2}}
  // overload-error@+1 {{'svhistseg' needs target feature sve2}}
  SVE_ACLE_FUNC(svhistseg,_s8,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqrdmulh_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh,_s8,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqrdmulh_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh,_n_s8,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svqdmulh_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh,_s8,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmulh_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh,_n_s8,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svsra_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svsra' needs target feature sve2}}
  SVE_ACLE_FUNC(svsra,_n_s8,,)(svundef_s8(), svundef_s8(), 1);
  // expected-error@+2 {{'svnbsl_s8' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svnbsl_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svqabs_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s8,_z,)(pg, svundef_s8());
  // expected-error@+2 {{'svqabs_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s8,_m,)(svundef_s8(), pg, svundef_s8());
  // expected-error@+2 {{'svqabs_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s8,_x,)(pg, svundef_s8());
  // expected-error@+2 {{'svcadd_s8' needs target feature sve2}}
  // overload-error@+1 {{'svcadd' needs target feature sve2}}
  SVE_ACLE_FUNC(svcadd,_s8,,)(svundef_s8(), svundef_s8(), 90);
  // expected-error@+2 {{'svtbl2_s8' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_s8,,)(svundef2_s8(), svundef_u8());
  // expected-error@+2 {{'svhsubr_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhsubr_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhsubr_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhsubr_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svhsubr_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svhsubr_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'sveortb_s8' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'sveortb_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svbcax_s8' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svbcax_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svqshlu_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshlu_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshlu,_n_s8,_z,)(pg, svundef_s8(), 1);
  // expected-error@+2 {{'svqrshl_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqrshl_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqrshl_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqrshl_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqrshl_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqrshl_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svcmla_s8' needs target feature sve2}}
  // overload-error@+1 {{'svcmla' needs target feature sve2}}
  SVE_ACLE_FUNC(svcmla,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8(), 90);
  // expected-error@+2 {{'svqsubr_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqsubr_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqsubr_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqsubr_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqsubr_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqsubr_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svrshr_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshr,_n_s8,_z,)(pg, svundef_s8(), 1);
  // expected-error@+2 {{'svaddp_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svaddp_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqadd_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqadd_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqadd_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqadd_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqadd_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqadd_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svtbx_s8' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_s8,,)(svundef_s8(), svundef_s8(), svundef_u8());
  // expected-error@+2 {{'svqrdcmlah_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqrdcmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8(), 90);
  // expected-error@+2 {{'svminp_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svminp_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqsub_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqsub_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqsub_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqsub_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqsub_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqsub_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svrsra_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svrsra' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsra,_n_s8,,)(svundef_s8(), svundef_s8(), 1);
  // expected-error@+2 {{'sveor3_s8' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'sveor3_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svhadd_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhadd_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svhadd_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhadd_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svhadd_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhadd_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqrdmlsh_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqrdmlsh_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svmaxp_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svmaxp_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svmatch_s8' needs target feature sve2}}
  // overload-error@+1 {{'svmatch' needs target feature sve2}}
  SVE_ACLE_FUNC(svmatch,_s8,,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svwhilerw_s8' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_s8,,)(const_i8_ptr, const_i8_ptr);
  // expected-error@+2 {{'svqcadd_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqcadd' needs target feature sve2}}
  SVE_ACLE_FUNC(svqcadd,_s8,,)(svundef_s8(), svundef_s8(), 90);
  // expected-error@+2 {{'svrhadd_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svrhadd_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svrhadd_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svrhadd_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svrhadd_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svrhadd_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svwhilewr_s8' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_s8,,)(const_i8_ptr, const_i8_ptr);
  // expected-error@+2 {{'svsli_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svsli' needs target feature sve2}}
  SVE_ACLE_FUNC(svsli,_n_s8,,)(svundef_s8(), svundef_s8(), 1);
  // expected-error@+2 {{'svnmatch_s8' needs target feature sve2}}
  // overload-error@+1 {{'svnmatch' needs target feature sve2}}
  SVE_ACLE_FUNC(svnmatch,_s8,,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svaba_s8' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svaba_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svuqadd_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s8,_m,)(pg, svundef_s8(), svundef_u8());
  // expected-error@+2 {{'svuqadd_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s8,_m,)(pg, svundef_s8(), u8);
  // expected-error@+2 {{'svuqadd_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s8,_z,)(pg, svundef_s8(), svundef_u8());
  // expected-error@+2 {{'svuqadd_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s8,_z,)(pg, svundef_s8(), u8);
  // expected-error@+2 {{'svuqadd_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s8,_x,)(pg, svundef_s8(), svundef_u8());
  // expected-error@+2 {{'svuqadd_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s8,_x,)(pg, svundef_s8(), u8);
  // expected-error@+2 {{'sveorbt_s8' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'sveorbt_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svbsl_s8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svbsl_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svhsub_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhsub_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhsub_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svhsub_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svhsub_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svhsub_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqrdmlah_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqrdmlah_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svbsl2n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svbsl2n_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svsri_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svsri' needs target feature sve2}}
  SVE_ACLE_FUNC(svsri,_n_s8,,)(svundef_s8(), svundef_s8(), 1);
  // expected-error@+2 {{'svbsl1n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svbsl1n_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{'svrshl_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svrshl_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svrshl_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svrshl_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svrshl_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svrshl_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqneg_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s8,_z,)(pg, svundef_s8());
  // expected-error@+2 {{'svqneg_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s8,_m,)(svundef_s8(), pg, svundef_s8());
  // expected-error@+2 {{'svqneg_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s8,_x,)(pg, svundef_s8());
  // expected-error@+2 {{'svxar_n_s8' needs target feature sve2}}
  // overload-error@+1 {{'svxar' needs target feature sve2}}
  SVE_ACLE_FUNC(svxar,_n_s8,,)(svundef_s8(), svundef_s8(), 1);
  // expected-error@+2 {{'svqshl_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqshl_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqshl_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqshl_n_s8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqshl_n_s8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{'svqshl_n_s8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s8,_x,)(pg, svundef_s8(), i8);

  // expected-error@+2 {{'svmullb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svmullb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svqrshrunb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrshrunb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshrunb,_n_s16,,)(svundef_s16(), 1);
  // expected-error@+2 {{'svqdmlalbt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalbt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmlalbt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalbt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svqrdmulh_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqrdmulh_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqrdmulh_lane_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh_lane,_s16,,)(svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svaddwb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_s16,,)(svundef_s16(), svundef_s8());
  // expected-error@+2 {{'svaddwb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_n_s16,,)(svundef_s16(), i8);
  // expected-error@+2 {{'svsubhnb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svsubhnb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqdmulh_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmulh_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqdmulh_lane_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh_lane,_s16,,)(svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svqshrunt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqshrunt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshrunt,_n_s16,,)(svundef_u8(), svundef_s16(), 1);
  // expected-error@+2 {{'svrsubhnt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_s16,,)(svundef_s8(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svrsubhnt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_n_s16,,)(svundef_s8(), svundef_s16(), i16);
  // expected-error@+2 {{'svnbsl_s16' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svnbsl_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svqdmlslb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmlslb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svsubhnt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_s16,,)(svundef_s8(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svsubhnt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_n_s16,,)(svundef_s8(), svundef_s16(), i16);
  // expected-error@+2 {{'svqabs_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s16,_z,)(pg, svundef_s16());
  // expected-error@+2 {{'svqabs_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s16,_m,)(svundef_s16(), pg, svundef_s16());
  // expected-error@+2 {{'svqabs_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s16,_x,)(pg, svundef_s16());
  // expected-error@+2 {{'svaddlbt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlbt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svaddlbt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlbt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svtbl2_s16' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_s16,,)(svundef2_s16(), svundef_u16());
  // expected-error@+2 {{'svshrnt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svshrnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), 1);
  // expected-error@+2 {{'svhsubr_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhsubr_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhsubr_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhsubr_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svhsubr_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svhsubr_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'sveortb_s16' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'sveortb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svqxtnb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnb,_s16,,)(svundef_s16());
  // expected-error@+2 {{'svmlalt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svmlalt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svshrnb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svshrnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svshrnb,_n_s16,,)(svundef_s16(), 1);
  // expected-error@+2 {{'svaddhnt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_s16,,)(svundef_s8(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svaddhnt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_n_s16,,)(svundef_s8(), svundef_s16(), i16);
  // expected-error@+2 {{'svmls_lane_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmls_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmls_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svqdmlalt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmlalt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svbcax_s16' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svbcax_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svqxtnt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnt,_s16,,)(svundef_s8(), svundef_s16());
  // expected-error@+2 {{'svqdmlalb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmlalb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svqrshl_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqrshl_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqrshl_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqrshl_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqrshl_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqrshl_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svsublbt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsublbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublbt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svsublbt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsublbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublbt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svqshrnt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqshrnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), 1);
  // expected-error@+2 {{'svqdmullt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmullt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svsublt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svsublt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svqdmlslbt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslbt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmlslbt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslbt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svadalp_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s16,_z,)(pg, svundef_s16(), svundef_s8());
  // expected-error@+2 {{'svadalp_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s16,_m,)(pg, svundef_s16(), svundef_s8());
  // expected-error@+2 {{'svadalp_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s16,_x,)(pg, svundef_s16(), svundef_s8());
  // expected-error@+2 {{'svmul_lane_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmul_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmul_lane,_s16,,)(svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svsubwt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_s16,,)(svundef_s16(), svundef_s8());
  // expected-error@+2 {{'svsubwt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_n_s16,,)(svundef_s16(), i8);
  // expected-error@+2 {{'svqsubr_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqsubr_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqsubr_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqsubr_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqsubr_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqsubr_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqrshrnt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrshrnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), 1);
  // expected-error@+2 {{'svaddp_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svaddp_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqadd_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqadd_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqadd_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqadd_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqadd_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqadd_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svabdlb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svabdlb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svtbx_s16' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_s16,,)(svundef_s16(), svundef_s16(), svundef_u16());
  // expected-error@+2 {{'svabdlt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svabdlt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svqrshrnb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrshrnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshrnb,_n_s16,,)(svundef_s16(), 1);
  // expected-error@+2 {{'svminp_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svminp_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqsub_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqsub_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqsub_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqsub_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqsub_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqsub_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svrsubhnb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svrsubhnb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svaddhnb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svaddhnb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svabalt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svabalt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svqshrnb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqshrnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshrnb,_n_s16,,)(svundef_s16(), 1);
  // expected-error@+2 {{'sveor3_s16' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'sveor3_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svhadd_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhadd_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svhadd_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhadd_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svhadd_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhadd_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqshrunb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqshrunb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshrunb,_n_s16,,)(svundef_s16(), 1);
  // expected-error@+2 {{'svmovlb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmovlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlb,_s16,,)(svundef_s8());
  // expected-error@+2 {{'svqrdmlsh_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqrdmlsh_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svqrdmlsh_lane_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svqdmlslt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmlslt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svmaxp_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svmaxp_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svmullt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svmullt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svmatch_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmatch' needs target feature sve2}}
  SVE_ACLE_FUNC(svmatch,_s16,,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqxtunb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqxtunb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtunb,_s16,,)(svundef_s16());
  // expected-error@+2 {{'svmla_lane_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmla_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmla_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svrshrnb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svrshrnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshrnb,_n_s16,,)(svundef_s16(), 1);
  // expected-error@+2 {{'svwhilerw_s16' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_s16,,)(const_i16_ptr, const_i16_ptr);
  // expected-error@+2 {{'svshllb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svshllb' needs target feature sve2}}
  SVE_ACLE_FUNC(svshllb,_n_s16,,)(svundef_s8(), 2);
  // expected-error@+2 {{'svrhadd_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svrhadd_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svrhadd_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svrhadd_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svrhadd_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svrhadd_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svraddhnb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svraddhnb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svwhilewr_s16' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_s16,,)(const_i16_ptr, const_i16_ptr);
  // expected-error@+2 {{'svmlalb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svmlalb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svsubwb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_s16,,)(svundef_s16(), svundef_s8());
  // expected-error@+2 {{'svsubwb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_n_s16,,)(svundef_s16(), i8);
  // expected-error@+2 {{'svnmatch_s16' needs target feature sve2}}
  // overload-error@+1 {{'svnmatch' needs target feature sve2}}
  SVE_ACLE_FUNC(svnmatch,_s16,,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svaba_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svaba_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svraddhnt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_s16,,)(svundef_s8(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svraddhnt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_n_s16,,)(svundef_s8(), svundef_s16(), i16);
  // expected-error@+2 {{'svuqadd_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s16,_m,)(pg, svundef_s16(), svundef_u16());
  // expected-error@+2 {{'svuqadd_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s16,_m,)(pg, svundef_s16(), u16);
  // expected-error@+2 {{'svuqadd_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s16,_z,)(pg, svundef_s16(), svundef_u16());
  // expected-error@+2 {{'svuqadd_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s16,_z,)(pg, svundef_s16(), u16);
  // expected-error@+2 {{'svuqadd_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s16,_x,)(pg, svundef_s16(), svundef_u16());
  // expected-error@+2 {{'svuqadd_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s16,_x,)(pg, svundef_s16(), u16);
  // expected-error@+2 {{'sveorbt_s16' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'sveorbt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svbsl_s16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svbsl_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svshllt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svshllt' needs target feature sve2}}
  SVE_ACLE_FUNC(svshllt,_n_s16,,)(svundef_s8(), 2);
  // expected-error@+2 {{'svsubltb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubltb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubltb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svsubltb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsubltb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubltb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svhsub_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhsub_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhsub_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svhsub_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svhsub_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svhsub_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svaddlb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svaddlb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svqrdmlah_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqrdmlah_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svqrdmlah_lane_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svqdmullb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svqdmullb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svbsl2n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svbsl2n_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svaddlt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svaddlt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svqxtunt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqxtunt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtunt,_s16,,)(svundef_u8(), svundef_s16());
  // expected-error@+2 {{'svqrshrunt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svqrshrunt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshrunt,_n_s16,,)(svundef_u8(), svundef_s16(), 1);
  // expected-error@+2 {{'svabalb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svabalb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svsublb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svsublb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{'svbsl1n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svbsl1n_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{'svrshl_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svrshl_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svrshl_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svrshl_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svrshl_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svrshl_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svaddwt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_s16,,)(svundef_s16(), svundef_s8());
  // expected-error@+2 {{'svaddwt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_n_s16,,)(svundef_s16(), i8);
  // expected-error@+2 {{'svmlslb_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svmlslb_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svmlslt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{'svmlslt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{'svqneg_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s16,_z,)(pg, svundef_s16());
  // expected-error@+2 {{'svqneg_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s16,_m,)(svundef_s16(), pg, svundef_s16());
  // expected-error@+2 {{'svqneg_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s16,_x,)(pg, svundef_s16());
  // expected-error@+2 {{'svmovlt_s16' needs target feature sve2}}
  // overload-error@+1 {{'svmovlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlt,_s16,,)(svundef_s8());
  // expected-error@+2 {{'svrshrnt_n_s16' needs target feature sve2}}
  // overload-error@+1 {{'svrshrnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), 1);
  // expected-error@+2 {{'svqshl_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqshl_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqshl_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqshl_n_s16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqshl_n_s16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{'svqshl_n_s16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s16,_x,)(pg, svundef_s16(), i16);

  // expected-error@+2 {{'svmullb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svmullb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svmullb_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmullb_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb_lane,_s32,,)(svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svqdmlalbt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalbt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmlalbt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalbt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svqrdmulh_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqrdmulh_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svaddwb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_s32,,)(svundef_s32(), svundef_s16());
  // expected-error@+2 {{'svaddwb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_n_s32,,)(svundef_s32(), i16);
  // expected-error@+2 {{'svsubhnb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svsubhnb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svqdmulh_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmulh_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svrsubhnt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_s32,,)(svundef_s16(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svrsubhnt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_n_s32,,)(svundef_s16(), svundef_s32(), i32);
  // expected-error@+2 {{'svnbsl_s32' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svnbsl_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svqdmlslb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmlslb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svqdmlslb_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslb_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svsubhnt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_s32,,)(svundef_s16(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svsubhnt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_n_s32,,)(svundef_s16(), svundef_s32(), i32);
  // expected-error@+2 {{'svqabs_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s32,_z,)(pg, svundef_s32());
  // expected-error@+2 {{'svqabs_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s32,_m,)(svundef_s32(), pg, svundef_s32());
  // expected-error@+2 {{'svqabs_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s32,_x,)(pg, svundef_s32());
  // expected-error@+2 {{'svwhilegt_b8_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b8' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b8,_s32,,)(i32, i32);
  // expected-error@+2 {{'svwhilegt_b16_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b16' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b16,_s32,,)(i32, i32);
  // expected-error@+2 {{'svwhilegt_b32_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b32' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b32,_s32,,)(i32, i32);
  // expected-error@+2 {{'svwhilegt_b64_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b64' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b64,_s32,,)(i32, i32);
  // expected-error@+2 {{'svaddlbt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlbt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svaddlbt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlbt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svtbl2_s32' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_s32,,)(svundef2_s32(), svundef_u32());
  // expected-error@+2 {{'svhsubr_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhsubr_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhsubr_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhsubr_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svhsubr_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svhsubr_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svhistcnt_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhistcnt_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhistcnt,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'sveortb_s32' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'sveortb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svqxtnb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnb,_s32,,)(svundef_s32());
  // expected-error@+2 {{'svmlalt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svmlalt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svmlalt_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svaddhnt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_s32,,)(svundef_s16(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svaddhnt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_n_s32,,)(svundef_s16(), svundef_s32(), i32);
  // expected-error@+2 {{'svldnt1uh_gather_u32base_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1uh_gather_u32offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u32, offset_s32, )(pg, const_u16_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1uh_gather_u32base_offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svldnt1uh_gather_u32base_index_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_index_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _index_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svqdmlalt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmlalt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svqdmlalt_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalt_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svbcax_s32' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svbcax_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svqxtnt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnt,_s32,,)(svundef_s16(), svundef_s32());
  // expected-error@+2 {{'svqdmlalb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmlalb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svqdmlalb_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalb_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svqrshl_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqrshl_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqrshl_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqrshl_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqrshl_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqrshl_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svcdot_s32' needs target feature sve2}}
  // overload-error@+1 {{'svcdot' needs target feature sve2}}
  SVE_ACLE_FUNC(svcdot,_s32,,)(svundef_s32(), svundef_s8(), svundef_s8(), 90);
  // expected-error@+2 {{'svsublbt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsublbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublbt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svsublbt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsublbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublbt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqdmullt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmullt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqdmullt_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullt_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullt_lane,_s32,,)(svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svsublt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svsublt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqdmlslbt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslbt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmlslbt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslbt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svadalp_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s32,_z,)(pg, svundef_s32(), svundef_s16());
  // expected-error@+2 {{'svadalp_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s32,_m,)(pg, svundef_s32(), svundef_s16());
  // expected-error@+2 {{'svadalp_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s32,_x,)(pg, svundef_s32(), svundef_s16());
  // expected-error@+2 {{'svwhilege_b8_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b8' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b8,_s32,,)(i32, i32);
  // expected-error@+2 {{'svwhilege_b16_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b16' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b16,_s32,,)(i32, i32);
  // expected-error@+2 {{'svwhilege_b32_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b32' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b32,_s32,,)(i32, i32);
  // expected-error@+2 {{'svwhilege_b64_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b64' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b64,_s32,,)(i32, i32);
  // expected-error@+2 {{'svsubwt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_s32,,)(svundef_s32(), svundef_s16());
  // expected-error@+2 {{'svsubwt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_n_s32,,)(svundef_s32(), i16);
  // expected-error@+2 {{'svqsubr_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqsubr_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqsubr_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqsubr_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqsubr_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqsubr_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svaddp_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svaddp_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqadd_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqadd_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqadd_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqadd_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqadd_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqadd_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svabdlb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svabdlb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svtbx_s32' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_s32,,)(svundef_s32(), svundef_s32(), svundef_u32());
  // expected-error@+2 {{'svabdlt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svabdlt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svminp_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svminp_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqsub_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqsub_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqsub_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqsub_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqsub_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqsub_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svrsubhnb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svrsubhnb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svaddhnb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svaddhnb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svabalt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svabalt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'sveor3_s32' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'sveor3_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svhadd_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhadd_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svhadd_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhadd_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svhadd_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhadd_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svmovlb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmovlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlb,_s32,,)(svundef_s16());
  // expected-error@+2 {{'svstnt1_scatter_u32base_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, , _s32)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svstnt1_scatter_u32offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u32, offset, _s32)(pg, i32_ptr, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svstnt1_scatter_u32base_offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _offset, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{'svstnt1_scatter_u32base_index_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _index, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{'svqrdmlsh_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqrdmlsh_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svqdmlslt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmlslt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svqdmlslt_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslt_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svmaxp_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svmaxp_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svmullt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svmullt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svmullt_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmullt_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt_lane,_s32,,)(svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svldnt1sh_gather_u32base_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1sh_gather_u32offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u32, offset_s32, )(pg, const_i16_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1sh_gather_u32base_offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svldnt1sh_gather_u32base_index_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_index_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _index_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svqxtunb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqxtunb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtunb,_s32,,)(svundef_s32());
  // expected-error@+2 {{'svwhilerw_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_s32,,)(const_i32_ptr, const_i32_ptr);
  // expected-error@+2 {{'svrhadd_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svrhadd_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svrhadd_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svrhadd_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svrhadd_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svrhadd_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svraddhnb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svraddhnb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svwhilewr_s32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_s32,,)(const_i32_ptr, const_i32_ptr);
  // expected-error@+2 {{'svmlalb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svmlalb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svmlalb_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svldnt1sb_gather_u32base_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1sb_gather_u32offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, u32, offset_s32, )(pg, const_i8_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1sb_gather_u32base_offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svsubwb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_s32,,)(svundef_s32(), svundef_s16());
  // expected-error@+2 {{'svsubwb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_n_s32,,)(svundef_s32(), i16);
  // expected-error@+2 {{'svldnt1ub_gather_u32base_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1ub_gather_u32offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, u32, offset_s32, )(pg, const_u8_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1ub_gather_u32base_offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svaba_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svaba_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svraddhnt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_s32,,)(svundef_s16(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svraddhnt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_n_s32,,)(svundef_s16(), svundef_s32(), i32);
  // expected-error@+2 {{'svuqadd_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s32,_m,)(pg, svundef_s32(), svundef_u32());
  // expected-error@+2 {{'svuqadd_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s32,_m,)(pg, svundef_s32(), u32);
  // expected-error@+2 {{'svuqadd_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s32,_z,)(pg, svundef_s32(), svundef_u32());
  // expected-error@+2 {{'svuqadd_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s32,_z,)(pg, svundef_s32(), u32);
  // expected-error@+2 {{'svuqadd_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s32,_x,)(pg, svundef_s32(), svundef_u32());
  // expected-error@+2 {{'svuqadd_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s32,_x,)(pg, svundef_s32(), u32);
  // expected-error@+2 {{'sveorbt_s32' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'sveorbt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svbsl_s32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svbsl_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svsubltb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubltb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubltb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svsubltb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsubltb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubltb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svhsub_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhsub_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhsub_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svhsub_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svhsub_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svhsub_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svldnt1_gather_u32base_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1_gather_u32offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _s32)(pg, const_i32_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1_gather_u32base_offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svldnt1_gather_u32base_index_s32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index_s32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svaddlb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svaddlb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqrdmlah_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqrdmlah_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svqdmullb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svqdmullb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqdmullb_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullb_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullb_lane,_s32,,)(svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svstnt1h_scatter_u32base_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, , _s32)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svstnt1h_scatter_u32offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u32, offset, _s32)(pg, i16_ptr, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svstnt1h_scatter_u32base_offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, _offset, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{'svstnt1h_scatter_u32base_index_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, _index, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{'svstnt1b_scatter_u32base_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u32base, , _s32)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svstnt1b_scatter_u32offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, u32, offset, _s32)(pg, i8_ptr, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svstnt1b_scatter_u32base_offset_s32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u32base, _offset, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{'svbsl2n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svbsl2n_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svaddlt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svaddlt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svqxtunt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svqxtunt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtunt,_s32,,)(svundef_u16(), svundef_s32());
  // expected-error@+2 {{'svabalb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svabalb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svsublb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svsublb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{'svbsl1n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svbsl1n_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{'svrshl_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svrshl_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svrshl_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svrshl_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svrshl_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svrshl_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svaddwt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_s32,,)(svundef_s32(), svundef_s16());
  // expected-error@+2 {{'svaddwt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_n_s32,,)(svundef_s32(), i16);
  // expected-error@+2 {{'svmlslb_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svmlslb_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svmlslb_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svmlslt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{'svmlslt_n_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{'svmlslt_lane_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt_lane' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), 1);
  // expected-error@+2 {{'svqneg_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s32,_z,)(pg, svundef_s32());
  // expected-error@+2 {{'svqneg_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s32,_m,)(svundef_s32(), pg, svundef_s32());
  // expected-error@+2 {{'svqneg_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s32,_x,)(pg, svundef_s32());
  // expected-error@+2 {{'svmovlt_s32' needs target feature sve2}}
  // overload-error@+1 {{'svmovlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlt,_s32,,)(svundef_s16());
  // expected-error@+2 {{'svqshl_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqshl_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqshl_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqshl_n_s32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqshl_n_s32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{'svqshl_n_s32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s32,_x,)(pg, svundef_s32(), i32);

  // expected-error@+2 {{'svmullb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svmullb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svqdmlalbt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalbt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmlalbt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalbt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svqrdmulh_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqrdmulh_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmulh,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{'svaddwb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_s64,,)(svundef_s64(), svundef_s32());
  // expected-error@+2 {{'svaddwb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_n_s64,,)(svundef_s64(), i32);
  // expected-error@+2 {{'svsubhnb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svsubhnb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{'svqdmulh_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqdmulh_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmulh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmulh,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{'svrsubhnt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_s64,,)(svundef_s32(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svrsubhnt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_n_s64,,)(svundef_s32(), svundef_s64(), i64);
  // expected-error@+2 {{'svnbsl_s64' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svnbsl_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svqdmlslb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmlslb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svsubhnt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_s64,,)(svundef_s32(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svsubhnt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_n_s64,,)(svundef_s32(), svundef_s64(), i64);
  // expected-error@+2 {{'svqabs_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s64,_z,)(pg, svundef_s64());
  // expected-error@+2 {{'svqabs_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s64,_m,)(svundef_s64(), pg, svundef_s64());
  // expected-error@+2 {{'svqabs_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqabs_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqabs,_s64,_x,)(pg, svundef_s64());
  // expected-error@+2 {{'svwhilegt_b8_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b8' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b8,_s64,,)(i64, i64);
  // expected-error@+2 {{'svwhilegt_b16_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b16' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b16,_s64,,)(i64, i64);
  // expected-error@+2 {{'svwhilegt_b32_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b32' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b32,_s64,,)(i64, i64);
  // expected-error@+2 {{'svwhilegt_b64_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b64' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b64,_s64,,)(i64, i64);
  // expected-error@+2 {{'svaddlbt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlbt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svaddlbt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlbt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svtbl2_s64' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_s64,,)(svundef2_s64(), svundef_u64());
  // expected-error@+2 {{'svhsubr_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhsubr_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhsubr_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhsubr_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svhsubr_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svhsubr_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svhistcnt_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhistcnt_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhistcnt,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'sveortb_s64' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'sveortb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svqxtnb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnb,_s64,,)(svundef_s64());
  // expected-error@+2 {{'svmlalt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svmlalt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svaddhnt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_s64,,)(svundef_s32(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svaddhnt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_n_s64,,)(svundef_s32(), svundef_s64(), i64);
  // expected-error@+2 {{'svldnt1uh_gather_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1uh_gather_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, s64, offset_s64, )(pg, const_u16_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1uh_gather_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u64, offset_s64, )(pg, const_u16_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1uh_gather_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1uh_gather_s64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, s64, index_s64, )(pg, const_u16_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1uh_gather_u64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u64, index_s64, )(pg, const_u16_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1uh_gather_u64base_index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svqdmlalt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmlalt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svbcax_s64' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svbcax_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svqxtnt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnt,_s64,,)(svundef_s32(), svundef_s64());
  // expected-error@+2 {{'svqdmlalb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmlalb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlalb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svqrshl_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqrshl_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqrshl_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqrshl_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqrshl_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqrshl_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svsublbt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsublbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublbt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svsublbt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsublbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublbt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svqdmullt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmullt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svsublt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svsublt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svqdmlslbt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslbt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmlslbt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslbt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslbt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svadalp_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s64,_z,)(pg, svundef_s64(), svundef_s32());
  // expected-error@+2 {{'svadalp_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s64,_m,)(pg, svundef_s64(), svundef_s32());
  // expected-error@+2 {{'svadalp_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_s64,_x,)(pg, svundef_s64(), svundef_s32());
  // expected-error@+2 {{'svwhilege_b8_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b8' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b8,_s64,,)(i64, i64);
  // expected-error@+2 {{'svwhilege_b16_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b16' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b16,_s64,,)(i64, i64);
  // expected-error@+2 {{'svwhilege_b32_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b32' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b32,_s64,,)(i64, i64);
  // expected-error@+2 {{'svwhilege_b64_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b64' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b64,_s64,,)(i64, i64);
  // expected-error@+2 {{'svsubwt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_s64,,)(svundef_s64(), svundef_s32());
  // expected-error@+2 {{'svsubwt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_n_s64,,)(svundef_s64(), i32);
  // expected-error@+2 {{'svqsubr_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqsubr_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqsubr_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqsubr_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqsubr_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqsubr_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svaddp_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svaddp_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqadd_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqadd_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqadd_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqadd_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqadd_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqadd_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svabdlb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svabdlb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svtbx_s64' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_s64,,)(svundef_s64(), svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svabdlt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svabdlt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svminp_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svminp_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqsub_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqsub_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqsub_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqsub_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqsub_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqsub_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svrsubhnb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svrsubhnb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{'svaddhnb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svaddhnb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{'svabalt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svabalt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'sveor3_s64' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'sveor3_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svhadd_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhadd_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svhadd_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhadd_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svhadd_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhadd_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svmovlb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmovlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlb,_s64,,)(svundef_s32());
  // expected-error@+2 {{'svstnt1_scatter_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, , _s64)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1_scatter_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, offset, _s64)(pg, i64_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svstnt1_scatter_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, offset, _s64)(pg, i64_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1_scatter_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _offset, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{'svstnt1_scatter_s64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, index, _s64)(pg, i64_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svstnt1_scatter_u64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, index, _s64)(pg, i64_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1_scatter_u64base_index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _index, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{'svqrdmlsh_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqrdmlsh_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlsh' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlsh,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svqdmlslt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmlslt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmlslt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svmaxp_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svmaxp_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svmullt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svmullt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svldnt1sh_gather_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1sh_gather_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, s64, offset_s64, )(pg, const_i16_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sh_gather_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u64, offset_s64, )(pg, const_i16_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sh_gather_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1sh_gather_s64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, s64, index_s64, )(pg, const_i16_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sh_gather_u64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u64, index_s64, )(pg, const_i16_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sh_gather_u64base_index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svqxtunb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqxtunb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtunb,_s64,,)(svundef_s64());
  // expected-error@+2 {{'svwhilerw_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_s64,,)(const_i64_ptr, const_i64_ptr);
  // expected-error@+2 {{'svrhadd_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svrhadd_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svrhadd_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svrhadd_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svrhadd_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svrhadd_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svraddhnb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svraddhnb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{'svwhilewr_s64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_s64,,)(const_i64_ptr, const_i64_ptr);
  // expected-error@+2 {{'svmlalb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svmlalb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svldnt1sb_gather_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1sb_gather_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, s64, offset_s64, )(pg, const_i8_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sb_gather_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, u64, offset_s64, )(pg, const_i8_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sb_gather_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svsubwb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_s64,,)(svundef_s64(), svundef_s32());
  // expected-error@+2 {{'svsubwb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_n_s64,,)(svundef_s64(), i32);
  // expected-error@+2 {{'svldnt1ub_gather_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1ub_gather_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, s64, offset_s64, )(pg, const_u8_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1ub_gather_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, u64, offset_s64, )(pg, const_u8_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1ub_gather_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svaba_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svaba_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svraddhnt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_s64,,)(svundef_s32(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svraddhnt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_n_s64,,)(svundef_s32(), svundef_s64(), i64);
  // expected-error@+2 {{'svuqadd_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s64,_m,)(pg, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svuqadd_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s64,_m,)(pg, svundef_s64(), u64);
  // expected-error@+2 {{'svuqadd_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s64,_z,)(pg, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svuqadd_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s64,_z,)(pg, svundef_s64(), u64);
  // expected-error@+2 {{'svuqadd_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_s64,_x,)(pg, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svuqadd_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svuqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svuqadd,_n_s64,_x,)(pg, svundef_s64(), u64);
  // expected-error@+2 {{'sveorbt_s64' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'sveorbt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svldnt1sw_gather_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1sw_gather_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, s64, offset_s64, )(pg, const_i32_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sw_gather_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, u64, offset_s64, )(pg, const_i32_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sw_gather_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1sw_gather_s64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, s64, index_s64, )(pg, const_i32_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sw_gather_u64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, u64, index_s64, )(pg, const_i32_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sw_gather_u64base_index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svbsl_s64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svbsl_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svsubltb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubltb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubltb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svsubltb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsubltb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubltb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svhsub_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhsub_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhsub_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svhsub_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svhsub_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svhsub_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svldnt1_gather_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _s64)(pg, const_i64_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1_gather_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _s64)(pg, const_i64_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1_gather_s64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _s64)(pg, const_i64_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1_gather_u64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _s64)(pg, const_i64_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_u64base_index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svaddlb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svaddlb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svqrdmlah_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqrdmlah_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqrdmlah' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrdmlah,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svqdmullb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svqdmullb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqdmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqdmullb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svldnt1uw_gather_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1uw_gather_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, s64, offset_s64, )(pg, const_u32_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1uw_gather_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, u64, offset_s64, )(pg, const_u32_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1uw_gather_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_offset_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1uw_gather_s64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, s64, index_s64, )(pg, const_u32_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1uw_gather_u64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, u64, index_s64, )(pg, const_u32_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1uw_gather_u64base_index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_index_s64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svstnt1h_scatter_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, , _s64)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1h_scatter_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, s64, offset, _s64)(pg, i16_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svstnt1h_scatter_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u64, offset, _s64)(pg, i16_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1h_scatter_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, _offset, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{'svstnt1h_scatter_s64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, s64, index, _s64)(pg, i16_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svstnt1h_scatter_u64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u64, index, _s64)(pg, i16_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1h_scatter_u64base_index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, _index, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{'svstnt1b_scatter_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u64base, , _s64)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1b_scatter_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, s64, offset, _s64)(pg, i8_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svstnt1b_scatter_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, u64, offset, _s64)(pg, i8_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1b_scatter_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u64base, _offset, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{'svbsl2n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svbsl2n_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svaddlt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svaddlt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svstnt1w_scatter_u64base_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, , _s64)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1w_scatter_s64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, s64, offset, _s64)(pg, i32_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svstnt1w_scatter_u64offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, u64, offset, _s64)(pg, i32_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1w_scatter_u64base_offset_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, _offset, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{'svstnt1w_scatter_s64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, s64, index, _s64)(pg, i32_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svstnt1w_scatter_u64index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, u64, index, _s64)(pg, i32_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svstnt1w_scatter_u64base_index_s64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, _index, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{'svqxtunt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svqxtunt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtunt,_s64,,)(svundef_u32(), svundef_s64());
  // expected-error@+2 {{'svabalb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svabalb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svsublb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svsublb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{'svbsl1n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svbsl1n_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{'svrshl_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svrshl_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svrshl_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svrshl_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svrshl_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svrshl_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svaddwt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_s64,,)(svundef_s64(), svundef_s32());
  // expected-error@+2 {{'svaddwt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_n_s64,,)(svundef_s64(), i32);
  // expected-error@+2 {{'svmlslb_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svmlslb_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svmlslt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{'svmlslt_n_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{'svqneg_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s64,_z,)(pg, svundef_s64());
  // expected-error@+2 {{'svqneg_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s64,_m,)(svundef_s64(), pg, svundef_s64());
  // expected-error@+2 {{'svqneg_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqneg_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqneg,_s64,_x,)(pg, svundef_s64());
  // expected-error@+2 {{'svmovlt_s64' needs target feature sve2}}
  // overload-error@+1 {{'svmovlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlt,_s64,,)(svundef_s32());
  // expected-error@+2 {{'svqshl_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqshl_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqshl_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{'svqshl_n_s64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqshl_n_s64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{'svqshl_n_s64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_s64,_x,)(pg, svundef_s64(), i64);

  // expected-error@+2 {{'svhistseg_u8' needs target feature sve2}}
  // overload-error@+1 {{'svhistseg' needs target feature sve2}}
  SVE_ACLE_FUNC(svhistseg,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svpmullb_pair_u8' needs target feature sve2}}
  // overload-error@+1 {{'svpmullb_pair' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullb_pair,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svpmullb_pair_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svpmullb_pair' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullb_pair,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svnbsl_u8' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svnbsl_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'svtbl2_u8' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_u8,,)(svundef2_u8(), svundef_u8());
  // expected-error@+2 {{'svhsubr_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhsubr_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhsubr_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhsubr_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svhsubr_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svhsubr_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svpmul_u8' needs target feature sve2}}
  // overload-error@+1 {{'svpmul' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmul,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svpmul_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svpmul' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmul,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{'sveortb_u8' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'sveortb_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'svbcax_u8' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svbcax_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'svqrshl_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u8,_z,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svqrshl_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u8,_m,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svqrshl_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u8,_x,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svqrshl_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u8,_z,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svqrshl_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u8,_m,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svqrshl_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u8,_x,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svpmullt_pair_u8' needs target feature sve2}}
  // overload-error@+1 {{'svpmullt_pair' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullt_pair,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svpmullt_pair_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svpmullt_pair' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullt_pair,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svqsubr_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqsubr_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqsubr_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqsubr_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svqsubr_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svqsubr_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svaddp_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svaddp_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqadd_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqadd_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svqadd_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqadd_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svqadd_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqadd_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svtbx_u8' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svminp_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svminp_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svsqadd_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u8,_m,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svsqadd_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u8,_m,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svsqadd_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u8,_z,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svsqadd_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u8,_z,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svsqadd_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u8,_x,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svsqadd_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u8,_x,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svqsub_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqsub_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqsub_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svqsub_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svqsub_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svqsub_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'sveor3_u8' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'sveor3_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'svhadd_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhadd_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svhadd_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhadd_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svhadd_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhadd_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svmaxp_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svmaxp_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svmatch_u8' needs target feature sve2}}
  // overload-error@+1 {{'svmatch' needs target feature sve2}}
  SVE_ACLE_FUNC(svmatch,_u8,,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svwhilerw_u8' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_u8,,)(const_u8_ptr, const_u8_ptr);
  // expected-error@+2 {{'svrhadd_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svrhadd_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svrhadd_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svrhadd_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svrhadd_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svrhadd_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svwhilewr_u8' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_u8,,)(const_u8_ptr, const_u8_ptr);
  // expected-error@+2 {{'svnmatch_u8' needs target feature sve2}}
  // overload-error@+1 {{'svnmatch' needs target feature sve2}}
  SVE_ACLE_FUNC(svnmatch,_u8,,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svaba_u8' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svaba_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'sveorbt_u8' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'sveorbt_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'svbsl_u8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svbsl_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'svhsub_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhsub_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhsub_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svhsub_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svhsub_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svhsub_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{'svbsl2n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svbsl2n_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'svbsl1n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svbsl1n_n_u8' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{'svrshl_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u8,_z,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svrshl_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u8,_m,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svrshl_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u8,_x,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svrshl_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u8,_z,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svrshl_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u8,_m,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svrshl_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u8,_x,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svqshl_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u8,_z,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svqshl_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u8,_m,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svqshl_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u8,_x,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{'svqshl_n_u8_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u8,_z,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svqshl_n_u8_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u8,_m,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{'svqshl_n_u8_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u8,_x,)(pg, svundef_u8(), i8);

  // expected-error@+2 {{'svmullb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svmullb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svpmullb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svpmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svpmullb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svpmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svaddwb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_u16,,)(svundef_u16(), svundef_u8());
  // expected-error@+2 {{'svaddwb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_n_u16,,)(svundef_u16(), u8);
  // expected-error@+2 {{'svsubhnb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svsubhnb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svrsubhnt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_u16,,)(svundef_u8(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svrsubhnt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_n_u16,,)(svundef_u8(), svundef_u16(), u16);
  // expected-error@+2 {{'svnbsl_u16' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svnbsl_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svsubhnt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_u16,,)(svundef_u8(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svsubhnt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_n_u16,,)(svundef_u8(), svundef_u16(), u16);
  // expected-error@+2 {{'svtbl2_u16' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_u16,,)(svundef2_u16(), svundef_u16());
  // expected-error@+2 {{'svhsubr_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhsubr_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhsubr_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhsubr_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svhsubr_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svhsubr_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'sveortb_u16' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'sveortb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svqxtnb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnb,_u16,,)(svundef_u16());
  // expected-error@+2 {{'svmlalt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svmlalt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{'svaddhnt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_u16,,)(svundef_u8(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svaddhnt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_n_u16,,)(svundef_u8(), svundef_u16(), u16);
  // expected-error@+2 {{'svbcax_u16' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svbcax_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svqxtnt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnt,_u16,,)(svundef_u8(), svundef_u16());
  // expected-error@+2 {{'svqrshl_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u16,_z,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svqrshl_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u16,_m,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svqrshl_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u16,_x,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svqrshl_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u16,_z,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svqrshl_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u16,_m,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svqrshl_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u16,_x,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svsublt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svsublt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svadalp_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u16,_z,)(pg, svundef_u16(), svundef_u8());
  // expected-error@+2 {{'svadalp_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u16,_m,)(pg, svundef_u16(), svundef_u8());
  // expected-error@+2 {{'svadalp_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u16,_x,)(pg, svundef_u16(), svundef_u8());
  // expected-error@+2 {{'svpmullt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svpmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svpmullt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svpmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svsubwt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_u16,,)(svundef_u16(), svundef_u8());
  // expected-error@+2 {{'svsubwt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_n_u16,,)(svundef_u16(), u8);
  // expected-error@+2 {{'svqsubr_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqsubr_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqsubr_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqsubr_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svqsubr_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svqsubr_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svaddp_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svaddp_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqadd_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqadd_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svqadd_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqadd_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svqadd_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqadd_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svabdlb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svabdlb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svtbx_u16' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svabdlt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svabdlt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svminp_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svminp_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svsqadd_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u16,_m,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svsqadd_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u16,_m,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svsqadd_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u16,_z,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svsqadd_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u16,_z,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svsqadd_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u16,_x,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svsqadd_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u16,_x,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svqsub_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqsub_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqsub_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svqsub_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svqsub_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svqsub_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svrsubhnb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svrsubhnb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svaddhnb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svaddhnb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svabalt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svabalt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{'sveor3_u16' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'sveor3_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svhadd_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhadd_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svhadd_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhadd_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svhadd_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhadd_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svmovlb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmovlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlb,_u16,,)(svundef_u8());
  // expected-error@+2 {{'svmaxp_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svmaxp_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svmullt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svmullt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svmatch_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmatch' needs target feature sve2}}
  SVE_ACLE_FUNC(svmatch,_u16,,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svwhilerw_u16' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_u16,,)(const_u16_ptr, const_u16_ptr);
  // expected-error@+2 {{'svrhadd_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svrhadd_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svrhadd_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svrhadd_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svrhadd_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svrhadd_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svraddhnb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svraddhnb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svwhilewr_u16' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_u16,,)(const_u16_ptr, const_u16_ptr);
  // expected-error@+2 {{'svmlalb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svmlalb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{'svsubwb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_u16,,)(svundef_u16(), svundef_u8());
  // expected-error@+2 {{'svsubwb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_n_u16,,)(svundef_u16(), u8);
  // expected-error@+2 {{'svnmatch_u16' needs target feature sve2}}
  // overload-error@+1 {{'svnmatch' needs target feature sve2}}
  SVE_ACLE_FUNC(svnmatch,_u16,,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svaba_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svaba_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svraddhnt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_u16,,)(svundef_u8(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svraddhnt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_n_u16,,)(svundef_u8(), svundef_u16(), u16);
  // expected-error@+2 {{'sveorbt_u16' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'sveorbt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svbsl_u16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svbsl_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svhsub_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhsub_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhsub_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svhsub_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svhsub_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svhsub_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{'svaddlb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svaddlb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svbsl2n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svbsl2n_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svaddlt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svaddlt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svabalb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svabalb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{'svsublb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svsublb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svbsl1n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svbsl1n_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{'svrshl_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u16,_z,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svrshl_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u16,_m,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svrshl_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u16,_x,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svrshl_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u16,_z,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svrshl_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u16,_m,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svrshl_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u16,_x,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svaddwt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_u16,,)(svundef_u16(), svundef_u8());
  // expected-error@+2 {{'svaddwt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_n_u16,,)(svundef_u16(), u8);
  // expected-error@+2 {{'svmlslb_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svmlslb_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{'svmlslt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svmlslt_n_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{'svmovlt_u16' needs target feature sve2}}
  // overload-error@+1 {{'svmovlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlt,_u16,,)(svundef_u8());
  // expected-error@+2 {{'svqshl_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u16,_z,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svqshl_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u16,_m,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svqshl_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u16,_x,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{'svqshl_n_u16_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u16,_z,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svqshl_n_u16_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u16,_m,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{'svqshl_n_u16_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u16,_x,)(pg, svundef_u16(), i16);

  // expected-error@+2 {{'svmullb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svmullb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svpmullb_pair_u32' needs target feature sve2}}
  // overload-error@+1 {{'svpmullb_pair' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullb_pair,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svpmullb_pair_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svpmullb_pair' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullb_pair,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svaddwb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_u32,,)(svundef_u32(), svundef_u16());
  // expected-error@+2 {{'svaddwb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_n_u32,,)(svundef_u32(), u16);
  // expected-error@+2 {{'svsubhnb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsubhnb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svrsubhnt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_u32,,)(svundef_u16(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svrsubhnt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_n_u32,,)(svundef_u16(), svundef_u32(), u32);
  // expected-error@+2 {{'svnbsl_u32' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svnbsl_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svsubhnt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_u32,,)(svundef_u16(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsubhnt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_n_u32,,)(svundef_u16(), svundef_u32(), u32);
  // expected-error@+2 {{'svwhilegt_b8_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b8' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b8,_u32,,)(u32, u32);
  // expected-error@+2 {{'svwhilegt_b16_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b16' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b16,_u32,,)(u32, u32);
  // expected-error@+2 {{'svwhilegt_b32_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b32' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b32,_u32,,)(u32, u32);
  // expected-error@+2 {{'svwhilegt_b64_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b64' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b64,_u32,,)(u32, u32);
  // expected-error@+2 {{'svtbl2_u32' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_u32,,)(svundef2_u32(), svundef_u32());
  // expected-error@+2 {{'svhsubr_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhsubr_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhsubr_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhsubr_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svhsubr_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svhsubr_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svhistcnt_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhistcnt_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhistcnt,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'sveortb_u32' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'sveortb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svqxtnb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnb,_u32,,)(svundef_u32());
  // expected-error@+2 {{'svmlalt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svmlalt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{'svaddhnt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_u32,,)(svundef_u16(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svaddhnt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_n_u32,,)(svundef_u16(), svundef_u32(), u32);
  // expected-error@+2 {{'svldnt1uh_gather_u32base_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1uh_gather_u32offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u32, offset_u32, )(pg, const_u16_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1uh_gather_u32base_offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svldnt1uh_gather_u32base_index_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_index_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _index_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svbcax_u32' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svbcax_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svqxtnt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnt,_u32,,)(svundef_u16(), svundef_u32());
  // expected-error@+2 {{'svqrshl_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u32,_z,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svqrshl_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u32,_m,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svqrshl_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u32,_x,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svqrshl_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u32,_z,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svqrshl_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u32,_m,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svqrshl_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u32,_x,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svsublt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svsublt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svadalp_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u32,_z,)(pg, svundef_u32(), svundef_u16());
  // expected-error@+2 {{'svadalp_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u32,_m,)(pg, svundef_u32(), svundef_u16());
  // expected-error@+2 {{'svadalp_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u32,_x,)(pg, svundef_u32(), svundef_u16());
  // expected-error@+2 {{'svwhilege_b8_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b8' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b8,_u32,,)(u32, u32);
  // expected-error@+2 {{'svwhilege_b16_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b16' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b16,_u32,,)(u32, u32);
  // expected-error@+2 {{'svwhilege_b32_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b32' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b32,_u32,,)(u32, u32);
  // expected-error@+2 {{'svwhilege_b64_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b64' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b64,_u32,,)(u32, u32);
  // expected-error@+2 {{'svpmullt_pair_u32' needs target feature sve2}}
  // overload-error@+1 {{'svpmullt_pair' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullt_pair,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svpmullt_pair_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svpmullt_pair' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullt_pair,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svsubwt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_u32,,)(svundef_u32(), svundef_u16());
  // expected-error@+2 {{'svsubwt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_n_u32,,)(svundef_u32(), u16);
  // expected-error@+2 {{'svqsubr_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqsubr_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqsubr_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqsubr_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svqsubr_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svqsubr_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svadclt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svadclt' needs target feature sve2}}
  SVE_ACLE_FUNC(svadclt,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svadclt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svadclt' needs target feature sve2}}
  SVE_ACLE_FUNC(svadclt,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svaddp_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svaddp_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svrecpe_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrecpe_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrecpe,_u32,_z,)(pg, svundef_u32());
  // expected-error@+2 {{'svrecpe_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrecpe_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrecpe,_u32,_m,)(svundef_u32(), pg, svundef_u32());
  // expected-error@+2 {{'svrecpe_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrecpe_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrecpe,_u32,_x,)(pg, svundef_u32());
  // expected-error@+2 {{'svqadd_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqadd_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svqadd_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqadd_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svqadd_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqadd_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svabdlb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svabdlb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svtbx_u32' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svabdlt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svabdlt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svminp_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svminp_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsqadd_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u32,_m,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svsqadd_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u32,_m,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svsqadd_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u32,_z,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svsqadd_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u32,_z,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svsqadd_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u32,_x,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svsqadd_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u32,_x,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svqsub_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqsub_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqsub_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svqsub_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svqsub_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svqsub_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svrsubhnb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svrsubhnb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svaddhnb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svaddhnb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svabalt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svabalt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{'sveor3_u32' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'sveor3_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svhadd_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhadd_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svhadd_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhadd_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svhadd_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhadd_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svmovlb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmovlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlb,_u32,,)(svundef_u16());
  // expected-error@+2 {{'svstnt1_scatter_u32base_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, , _u32)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svstnt1_scatter_u32offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u32, offset, _u32)(pg, u32_ptr, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svstnt1_scatter_u32base_offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _offset, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{'svstnt1_scatter_u32base_index_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _index, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{'svmaxp_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svmaxp_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsbclt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsbclt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsbclt,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsbclt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsbclt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsbclt,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svmullt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svmullt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svldnt1sh_gather_u32base_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1sh_gather_u32offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u32, offset_u32, )(pg, const_i16_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1sh_gather_u32base_offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svldnt1sh_gather_u32base_index_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_index_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _index_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svwhilerw_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_u32,,)(const_u32_ptr, const_u32_ptr);
  // expected-error@+2 {{'svrhadd_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svrhadd_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svrhadd_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svrhadd_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svrhadd_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svrhadd_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svraddhnb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svraddhnb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svwhilewr_u32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_u32,,)(const_u32_ptr, const_u32_ptr);
  // expected-error@+2 {{'svmlalb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svmlalb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{'svldnt1sb_gather_u32base_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1sb_gather_u32offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, u32, offset_u32, )(pg, const_i8_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1sb_gather_u32base_offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svsubwb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_u32,,)(svundef_u32(), svundef_u16());
  // expected-error@+2 {{'svsubwb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_n_u32,,)(svundef_u32(), u16);
  // expected-error@+2 {{'svldnt1ub_gather_u32base_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1ub_gather_u32offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, u32, offset_u32, )(pg, const_u8_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1ub_gather_u32base_offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svaba_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svaba_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svraddhnt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_u32,,)(svundef_u16(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svraddhnt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_n_u32,,)(svundef_u16(), svundef_u32(), u32);
  // expected-error@+2 {{'sveorbt_u32' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'sveorbt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svbsl_u32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svbsl_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svadclb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svadclb' needs target feature sve2}}
  SVE_ACLE_FUNC(svadclb,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svadclb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svadclb' needs target feature sve2}}
  SVE_ACLE_FUNC(svadclb,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svhsub_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhsub_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhsub_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svhsub_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svhsub_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svhsub_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{'svldnt1_gather_u32base_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1_gather_u32offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _u32)(pg, const_u32_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1_gather_u32base_offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svldnt1_gather_u32base_index_u32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index_u32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svaddlb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svaddlb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svstnt1h_scatter_u32base_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, , _u32)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svstnt1h_scatter_u32offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u32, offset, _u32)(pg, u16_ptr, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svstnt1h_scatter_u32base_offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, _offset, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{'svstnt1h_scatter_u32base_index_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, _index, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{'svstnt1b_scatter_u32base_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u32base, , _u32)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svstnt1b_scatter_u32offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, u32, offset, _u32)(pg, u8_ptr, svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svstnt1b_scatter_u32base_offset_u32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u32base, _offset, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{'svbsl2n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svbsl2n_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svaddlt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svaddlt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svabalb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svabalb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{'svsublb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svsublb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svsbclb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsbclb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsbclb,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsbclb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svsbclb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsbclb,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svbsl1n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svbsl1n_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{'svrshl_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u32,_z,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svrshl_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u32,_m,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svrshl_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u32,_x,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svrshl_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u32,_z,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svrshl_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u32,_m,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svrshl_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u32,_x,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svrsqrte_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svrsqrte_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsqrte,_u32,_z,)(pg, svundef_u32());
  // expected-error@+2 {{'svrsqrte_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svrsqrte_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsqrte,_u32,_m,)(svundef_u32(), pg, svundef_u32());
  // expected-error@+2 {{'svrsqrte_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svrsqrte_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsqrte,_u32,_x,)(pg, svundef_u32());
  // expected-error@+2 {{'svaddwt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_u32,,)(svundef_u32(), svundef_u16());
  // expected-error@+2 {{'svaddwt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_n_u32,,)(svundef_u32(), u16);
  // expected-error@+2 {{'svmlslb_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svmlslb_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{'svmlslt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svmlslt_n_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{'svmovlt_u32' needs target feature sve2}}
  // overload-error@+1 {{'svmovlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlt,_u32,,)(svundef_u16());
  // expected-error@+2 {{'svqshl_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u32,_z,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svqshl_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u32,_m,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svqshl_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u32,_x,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{'svqshl_n_u32_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u32,_z,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svqshl_n_u32_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u32,_m,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{'svqshl_n_u32_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u32,_x,)(pg, svundef_u32(), i32);

  // expected-error@+2 {{'svmullb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svmullb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svpmullb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svpmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svpmullb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svpmullb' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svaddwb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_u64,,)(svundef_u64(), svundef_u32());
  // expected-error@+2 {{'svaddwb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwb,_n_u64,,)(svundef_u64(), u32);
  // expected-error@+2 {{'svsubhnb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svsubhnb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnb,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svrsubhnt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_u64,,)(svundef_u32(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svrsubhnt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnt,_n_u64,,)(svundef_u32(), svundef_u64(), u64);
  // expected-error@+2 {{'svnbsl_u64' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svnbsl_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svnbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svnbsl,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svsubhnt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_u64,,)(svundef_u32(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svsubhnt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsubhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubhnt,_n_u64,,)(svundef_u32(), svundef_u64(), u64);
  // expected-error@+2 {{'svwhilegt_b8_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b8' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b8,_u64,,)(u64, u64);
  // expected-error@+2 {{'svwhilegt_b16_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b16' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b16,_u64,,)(u64, u64);
  // expected-error@+2 {{'svwhilegt_b32_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b32' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b32,_u64,,)(u64, u64);
  // expected-error@+2 {{'svwhilegt_b64_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilegt_b64' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilegt_b64,_u64,,)(u64, u64);
  // expected-error@+2 {{'svtbl2_u64' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_u64,,)(svundef2_u64(), svundef_u64());
  // expected-error@+2 {{'svhsubr_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhsubr_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhsubr_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhsubr_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svhsubr_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svhsubr_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsubr,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svhistcnt_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhistcnt_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhistcnt,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'sveortb_u64' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'sveortb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'sveortb' needs target feature sve2}}
  SVE_ACLE_FUNC(sveortb,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svqxtnb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnb,_u64,,)(svundef_u64());
  // expected-error@+2 {{'svmlalt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svmlalt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmlalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalt,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{'svaddhnt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_u64,,)(svundef_u32(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svaddhnt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnt,_n_u64,,)(svundef_u32(), svundef_u64(), u64);
  // expected-error@+2 {{'svldnt1uh_gather_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1uh_gather_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, s64, offset_u64, )(pg, const_u16_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1uh_gather_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u64, offset_u64, )(pg, const_u16_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1uh_gather_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1uh_gather_s64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, s64, index_u64, )(pg, const_u16_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1uh_gather_u64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u64, index_u64, )(pg, const_u16_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1uh_gather_u64base_index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uh_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svbcax_u64' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svbcax_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svbcax' needs target feature sve2}}
  SVE_ACLE_FUNC(svbcax,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svqxtnt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svqxtnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svqxtnt,_u64,,)(svundef_u32(), svundef_u64());
  // expected-error@+2 {{'svqrshl_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u64,_z,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svqrshl_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u64,_m,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svqrshl_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_u64,_x,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svqrshl_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u64,_z,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svqrshl_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u64,_m,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svqrshl_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqrshl,_n_u64,_x,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svsublt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsublt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsublt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svadalp_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u64,_z,)(pg, svundef_u64(), svundef_u32());
  // expected-error@+2 {{'svadalp_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u64,_m,)(pg, svundef_u64(), svundef_u32());
  // expected-error@+2 {{'svadalp_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svadalp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svadalp,_u64,_x,)(pg, svundef_u64(), svundef_u32());
  // expected-error@+2 {{'svwhilege_b8_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b8' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b8,_u64,,)(u64, u64);
  // expected-error@+2 {{'svwhilege_b16_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b16' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b16,_u64,,)(u64, u64);
  // expected-error@+2 {{'svwhilege_b32_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b32' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b32,_u64,,)(u64, u64);
  // expected-error@+2 {{'svwhilege_b64_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilege_b64' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilege_b64,_u64,,)(u64, u64);
  // expected-error@+2 {{'svpmullt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svpmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svpmullt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svpmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svpmullt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svsubwt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_u64,,)(svundef_u64(), svundef_u32());
  // expected-error@+2 {{'svsubwt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsubwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwt,_n_u64,,)(svundef_u64(), u32);
  // expected-error@+2 {{'svqsubr_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqsubr_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqsubr_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqsubr_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svqsubr_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svqsubr_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsubr_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsubr,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svadclt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svadclt' needs target feature sve2}}
  SVE_ACLE_FUNC(svadclt,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svadclt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svadclt' needs target feature sve2}}
  SVE_ACLE_FUNC(svadclt,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svaddp_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svaddp_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqadd_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqadd_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svqadd_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqadd_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svqadd_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqadd_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqadd,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svabdlb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svabdlb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svabdlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svtbx_u64' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svabdlt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svabdlt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svabdlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabdlt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svminp_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svminp_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svsqadd_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u64,_m,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svsqadd_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u64,_m,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svsqadd_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u64,_z,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svsqadd_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u64,_z,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svsqadd_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_u64,_x,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svsqadd_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svsqadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svsqadd,_n_u64,_x,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svqsub_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqsub_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqsub_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svqsub_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svqsub_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svqsub_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqsub,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svrsubhnb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svrsubhnb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svrsubhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svrsubhnb,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svaddhnb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svaddhnb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddhnb,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svabalt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svabalt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svabalt' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalt,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{'sveor3_u64' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'sveor3_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'sveor3' needs target feature sve2}}
  SVE_ACLE_FUNC(sveor3,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svhadd_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhadd_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svhadd_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhadd_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svhadd_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhadd_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhadd,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svmovlb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmovlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlb,_u64,,)(svundef_u32());
  // expected-error@+2 {{'svstnt1_scatter_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, , _u64)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1_scatter_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, offset, _u64)(pg, u64_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svstnt1_scatter_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, offset, _u64)(pg, u64_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1_scatter_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _offset, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{'svstnt1_scatter_s64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, index, _u64)(pg, u64_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svstnt1_scatter_u64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, index, _u64)(pg, u64_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1_scatter_u64base_index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _index, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{'svmaxp_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svmaxp_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svsbclt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsbclt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsbclt,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svsbclt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsbclt' needs target feature sve2}}
  SVE_ACLE_FUNC(svsbclt,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svmullt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svmullt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmullt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmullt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svldnt1sh_gather_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1sh_gather_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, s64, offset_u64, )(pg, const_i16_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sh_gather_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u64, offset_u64, )(pg, const_i16_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sh_gather_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1sh_gather_s64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, s64, index_u64, )(pg, const_i16_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sh_gather_u64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u64, index_u64, )(pg, const_i16_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sh_gather_u64base_index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sh_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svwhilerw_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_u64,,)(const_u64_ptr, const_u64_ptr);
  // expected-error@+2 {{'svrhadd_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svrhadd_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svrhadd_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svrhadd_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svrhadd_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svrhadd_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svrhadd_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrhadd,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svraddhnb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svraddhnb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnb' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnb,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svwhilewr_u64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_u64,,)(const_u64_ptr, const_u64_ptr);
  // expected-error@+2 {{'svmlalb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svmlalb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmlalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlalb,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{'svldnt1sb_gather_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1sb_gather_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, s64, offset_u64, )(pg, const_i8_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sb_gather_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, u64, offset_u64, )(pg, const_i8_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sb_gather_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sb_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svsubwb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_u64,,)(svundef_u64(), svundef_u32());
  // expected-error@+2 {{'svsubwb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsubwb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsubwb,_n_u64,,)(svundef_u64(), u32);
  // expected-error@+2 {{'svldnt1ub_gather_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1ub_gather_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, s64, offset_u64, )(pg, const_u8_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1ub_gather_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, u64, offset_u64, )(pg, const_u8_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1ub_gather_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1ub_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svaba_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svaba_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaba' needs target feature sve2}}
  SVE_ACLE_FUNC(svaba,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svraddhnt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_u64,,)(svundef_u32(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svraddhnt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svraddhnt' needs target feature sve2}}
  SVE_ACLE_FUNC(svraddhnt,_n_u64,,)(svundef_u32(), svundef_u64(), u64);
  // expected-error@+2 {{'sveorbt_u64' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'sveorbt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'sveorbt' needs target feature sve2}}
  SVE_ACLE_FUNC(sveorbt,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svldnt1sw_gather_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1sw_gather_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, s64, offset_u64, )(pg, const_i32_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sw_gather_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, u64, offset_u64, )(pg, const_i32_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sw_gather_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1sw_gather_s64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, s64, index_u64, )(pg, const_i32_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1sw_gather_u64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, u64, index_u64, )(pg, const_i32_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1sw_gather_u64base_index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1sw_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svbsl_u64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svbsl_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svadclb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svadclb' needs target feature sve2}}
  SVE_ACLE_FUNC(svadclb,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svadclb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svadclb' needs target feature sve2}}
  SVE_ACLE_FUNC(svadclb,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svhsub_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhsub_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhsub_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svhsub_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svhsub_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svhsub_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svhsub_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svhsub,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{'svldnt1_gather_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _u64)(pg, const_u64_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1_gather_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _u64)(pg, const_u64_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1_gather_s64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _u64)(pg, const_u64_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1_gather_u64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _u64)(pg, const_u64_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_u64base_index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svaddlb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svaddlb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlb' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svldnt1uw_gather_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1uw_gather_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, s64, offset_u64, )(pg, const_u32_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1uw_gather_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, u64, offset_u64, )(pg, const_u32_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1uw_gather_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_offset_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1uw_gather_s64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, s64, index_u64, )(pg, const_u32_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1uw_gather_u64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, u64, index_u64, )(pg, const_u32_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1uw_gather_u64base_index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1uw_gather_index_u64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svstnt1h_scatter_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, , _u64)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1h_scatter_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, s64, offset, _u64)(pg, u16_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svstnt1h_scatter_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u64, offset, _u64)(pg, u16_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1h_scatter_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, _offset, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{'svstnt1h_scatter_s64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, s64, index, _u64)(pg, u16_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svstnt1h_scatter_u64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u64, index, _u64)(pg, u16_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1h_scatter_u64base_index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1h_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, _index, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{'svstnt1b_scatter_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u64base, , _u64)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1b_scatter_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, s64, offset, _u64)(pg, u8_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svstnt1b_scatter_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, u64, offset, _u64)(pg, u8_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1b_scatter_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1b_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u64base, _offset, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{'svbsl2n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svbsl2n_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl2n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl2n,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svaddlt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svaddlt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddlt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svstnt1w_scatter_u64base_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, , _u64)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1w_scatter_s64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, s64, offset, _u64)(pg, u32_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svstnt1w_scatter_u64offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, u64, offset, _u64)(pg, u32_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1w_scatter_u64base_offset_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, _offset, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{'svstnt1w_scatter_s64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, s64, index, _u64)(pg, u32_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{'svstnt1w_scatter_u64index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, u64, index, _u64)(pg, u32_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svstnt1w_scatter_u64base_index_u64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1w_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, _index, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{'svabalb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svabalb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svabalb' needs target feature sve2}}
  SVE_ACLE_FUNC(svabalb,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{'svsublb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsublb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsublb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsublb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svsbclb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsbclb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsbclb,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svsbclb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svsbclb' needs target feature sve2}}
  SVE_ACLE_FUNC(svsbclb,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svbsl1n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svbsl1n_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svbsl1n' needs target feature sve2}}
  SVE_ACLE_FUNC(svbsl1n,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{'svrshl_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u64,_z,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svrshl_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u64,_m,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svrshl_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_u64,_x,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svrshl_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u64,_z,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svrshl_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u64,_m,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svrshl_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svrshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svrshl,_n_u64,_x,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svaddwt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_u64,,)(svundef_u64(), svundef_u32());
  // expected-error@+2 {{'svaddwt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svaddwt' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddwt,_n_u64,,)(svundef_u64(), u32);
  // expected-error@+2 {{'svmlslb_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svmlslb_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmlslb' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslb,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{'svmlslt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svmlslt_n_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmlslt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmlslt,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{'svmovlt_u64' needs target feature sve2}}
  // overload-error@+1 {{'svmovlt' needs target feature sve2}}
  SVE_ACLE_FUNC(svmovlt,_u64,,)(svundef_u32());
  // expected-error@+2 {{'svqshl_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u64,_z,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svqshl_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u64,_m,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svqshl_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_u64,_x,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{'svqshl_n_u64_z' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u64,_z,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svqshl_n_u64_m' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u64,_m,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svqshl_n_u64_x' needs target feature sve2}}
  // overload-error@+1 {{'svqshl_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svqshl,_n_u64,_x,)(pg, svundef_u64(), i64);

  // expected-error@+2 {{'svlogb_f16_z' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f16,_z,)(pg, svundef_f16());
  // expected-error@+2 {{'svlogb_f16_m' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f16,_m,)(svundef_s16(), pg, svundef_f16());
  // expected-error@+2 {{'svlogb_f16_x' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f16,_x,)(pg, svundef_f16());
  // expected-error@+2 {{'svminnmp_f16_m' needs target feature sve2}}
  // overload-error@+1 {{'svminnmp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminnmp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svminnmp_f16_x' needs target feature sve2}}
  // overload-error@+1 {{'svminnmp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminnmp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svtbl2_f16' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_f16,,)(svundef2_f16(), svundef_u16());
  // expected-error@+2 {{'svaddp_f16_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svaddp_f16_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svtbx_f16' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_f16,,)(svundef_f16(), svundef_f16(), svundef_u16());
  // expected-error@+2 {{'svminp_f16_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svminp_f16_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxp_f16_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxp_f16_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxnmp_f16_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxnmp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxnmp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svmaxnmp_f16_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxnmp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxnmp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{'svwhilerw_f16' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_f16,,)(const_f16_ptr, const_f16_ptr);
  // expected-error@+2 {{'svwhilewr_f16' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_f16,,)(const_f16_ptr, const_f16_ptr);
  // expected-error@+2 {{'svcvtlt_f32_f16_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtlt_f32_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtlt_f32,_f16,_m,)(svundef_f32(), pg, svundef_f16());
  // expected-error@+2 {{'svcvtlt_f32_f16_x' needs target feature sve2}}
  // overload-error@+1 {{'svcvtlt_f32_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtlt_f32,_f16,_x,)(pg, svundef_f16());

  // expected-error@+2 {{'svlogb_f32_z' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f32,_z,)(pg, svundef_f32());
  // expected-error@+2 {{'svlogb_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f32,_m,)(svundef_s32(), pg, svundef_f32());
  // expected-error@+2 {{'svlogb_f32_x' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f32,_x,)(pg, svundef_f32());
  // expected-error@+2 {{'svminnmp_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svminnmp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminnmp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svminnmp_f32_x' needs target feature sve2}}
  // overload-error@+1 {{'svminnmp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminnmp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svtbl2_f32' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_f32,,)(svundef2_f32(), svundef_u32());
  // expected-error@+2 {{'svaddp_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svaddp_f32_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svtbx_f32' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_f32,,)(svundef_f32(), svundef_f32(), svundef_u32());
  // expected-error@+2 {{'svminp_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svminp_f32_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svstnt1_scatter_u32base_f32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, , _f32)(pg, svundef_u32(), svundef_f32());
  // expected-error@+2 {{'svstnt1_scatter_u32offset_f32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u32, offset, _f32)(pg, f32_ptr, svundef_u32(), svundef_f32());
  // expected-error@+2 {{'svstnt1_scatter_u32base_offset_f32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _offset, _f32)(pg, svundef_u32(), i64, svundef_f32());
  // expected-error@+2 {{'svstnt1_scatter_u32base_index_f32' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _index, _f32)(pg, svundef_u32(), i64, svundef_f32());
  // expected-error@+2 {{'svmaxp_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxp_f32_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxnmp_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxnmp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxnmp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svmaxnmp_f32_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxnmp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxnmp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{'svwhilerw_f32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_f32,,)(const_f32_ptr, const_f32_ptr);
  // expected-error@+2 {{'svcvtnt_f16_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtnt_f16_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtnt_f16,_f32,_m,)(svundef_f16(), pg, svundef_f32());
  // expected-error@+2 {{'svcvtnt_f16_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtnt_f16_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtnt_f16,_f32,_x,)(svundef_f16(), pg, svundef_f32());
  // expected-error@+2 {{'svwhilewr_f32' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_f32,,)(const_f32_ptr, const_f32_ptr);
  // expected-error@+2 {{'svcvtlt_f64_f32_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtlt_f64_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtlt_f64,_f32,_m,)(svundef_f64(), pg, svundef_f32());
  // expected-error@+2 {{'svcvtlt_f64_f32_x' needs target feature sve2}}
  // overload-error@+1 {{'svcvtlt_f64_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtlt_f64,_f32,_x,)(pg, svundef_f32());
  // expected-error@+2 {{'svldnt1_gather_u32base_f32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_f32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _f32, )(pg, svundef_u32());
  // expected-error@+2 {{'svldnt1_gather_u32offset_f32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _f32)(pg, const_f32_ptr, svundef_u32());
  // expected-error@+2 {{'svldnt1_gather_u32base_offset_f32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset_f32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_f32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{'svldnt1_gather_u32base_index_f32' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index_f32' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_f32, )(pg, svundef_u32(), i64);

  // expected-error@+2 {{'svlogb_f64_z' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f64,_z,)(pg, svundef_f64());
  // expected-error@+2 {{'svlogb_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f64,_m,)(svundef_s64(), pg, svundef_f64());
  // expected-error@+2 {{'svlogb_f64_x' needs target feature sve2}}
  // overload-error@+1 {{'svlogb_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svlogb,_f64,_x,)(pg, svundef_f64());
  // expected-error@+2 {{'svminnmp_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svminnmp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminnmp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svminnmp_f64_x' needs target feature sve2}}
  // overload-error@+1 {{'svminnmp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminnmp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svtbl2_f64' needs target feature sve2}}
  // overload-error@+1 {{'svtbl2' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbl2,_f64,,)(svundef2_f64(), svundef_u64());
  // expected-error@+2 {{'svaddp_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svaddp_f64_x' needs target feature sve2}}
  // overload-error@+1 {{'svaddp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svaddp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svtbx_f64' needs target feature sve2}}
  // overload-error@+1 {{'svtbx' needs target feature sve2}}
  SVE_ACLE_FUNC(svtbx,_f64,,)(svundef_f64(), svundef_f64(), svundef_u64());
  // expected-error@+2 {{'svminp_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svminp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svminp_f64_x' needs target feature sve2}}
  // overload-error@+1 {{'svminp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svminp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svstnt1_scatter_u64base_f64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, , _f64)(pg, svundef_u64(), svundef_f64());
  // expected-error@+2 {{'svstnt1_scatter_s64offset_f64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, offset, _f64)(pg, f64_ptr, svundef_s64(), svundef_f64());
  // expected-error@+2 {{'svstnt1_scatter_u64offset_f64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, offset, _f64)(pg, f64_ptr, svundef_u64(), svundef_f64());
  // expected-error@+2 {{'svstnt1_scatter_u64base_offset_f64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _offset, _f64)(pg, svundef_u64(), i64, svundef_f64());
  // expected-error@+2 {{'svstnt1_scatter_s64index_f64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, index, _f64)(pg, f64_ptr, svundef_s64(), svundef_f64());
  // expected-error@+2 {{'svstnt1_scatter_u64index_f64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, index, _f64)(pg, f64_ptr, svundef_u64(), svundef_f64());
  // expected-error@+2 {{'svstnt1_scatter_u64base_index_f64' needs target feature sve2}}
  // overload-error@+1 {{'svstnt1_scatter_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _index, _f64)(pg, svundef_u64(), i64, svundef_f64());
  // expected-error@+2 {{'svmaxp_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxp_f64_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxnmp_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svmaxnmp_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxnmp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svmaxnmp_f64_x' needs target feature sve2}}
  // overload-error@+1 {{'svmaxnmp_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svmaxnmp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{'svwhilerw_f64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilerw' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilerw,_f64,,)(const_f64_ptr, const_f64_ptr);
  // expected-error@+2 {{'svcvtnt_f32_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtnt_f32_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtnt_f32,_f64,_m,)(svundef_f32(), pg, svundef_f64());
  // expected-error@+2 {{'svcvtnt_f32_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtnt_f32_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtnt_f32,_f64,_x,)(svundef_f32(), pg, svundef_f64());
  // expected-error@+2 {{'svwhilewr_f64' needs target feature sve2}}
  // overload-error@+1 {{'svwhilewr' needs target feature sve2}}
  SVE_ACLE_FUNC(svwhilewr,_f64,,)(const_f64_ptr, const_f64_ptr);
  // expected-error@+2 {{'svcvtx_f32_f64_z' needs target feature sve2}}
  // overload-error@+1 {{'svcvtx_f32_z' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtx_f32,_f64,_z,)(pg, svundef_f64());
  // expected-error@+2 {{'svcvtx_f32_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtx_f32_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtx_f32,_f64,_m,)(svundef_f32(), pg, svundef_f64());
  // expected-error@+2 {{'svcvtx_f32_f64_x' needs target feature sve2}}
  // overload-error@+1 {{'svcvtx_f32_x' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtx_f32,_f64,_x,)(pg, svundef_f64());
  // expected-error@+2 {{'svldnt1_gather_u64base_f64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_f64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _f64, )(pg, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_s64offset_f64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _f64)(pg, const_f64_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1_gather_u64offset_f64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _f64)(pg, const_f64_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_u64base_offset_f64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_offset_f64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_f64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svldnt1_gather_s64index_f64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _f64)(pg, const_f64_ptr, svundef_s64());
  // expected-error@+2 {{'svldnt1_gather_u64index_f64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _f64)(pg, const_f64_ptr, svundef_u64());
  // expected-error@+2 {{'svldnt1_gather_u64base_index_f64' needs target feature sve2}}
  // overload-error@+1 {{'svldnt1_gather_index_f64' needs target feature sve2}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_f64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{'svcvtxnt_f32_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtxnt_f32_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtxnt_f32,_f64,_m,)(svundef_f32(), pg, svundef_f64());
  // expected-error@+2 {{'svcvtxnt_f32_f64_m' needs target feature sve2}}
  // overload-error@+1 {{'svcvtxnt_f32_m' needs target feature sve2}}
  SVE_ACLE_FUNC(svcvtxnt_f32,_f64,_x,)(svundef_f32(), pg, svundef_f64());
}
