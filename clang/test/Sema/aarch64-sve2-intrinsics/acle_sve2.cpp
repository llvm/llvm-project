// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error,note %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error,note %s

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

void test_s8(svbool_t pg, const int8_t *const_i8_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svhistseg_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svhistseg'}}
  SVE_ACLE_FUNC(svhistseg,_s8,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh'}}
  SVE_ACLE_FUNC(svqrdmulh,_s8,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh'}}
  SVE_ACLE_FUNC(svqrdmulh,_n_s8,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh'}}
  SVE_ACLE_FUNC(svqdmulh,_s8,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh'}}
  SVE_ACLE_FUNC(svqdmulh,_n_s8,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsra_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svsra'}}
  SVE_ACLE_FUNC(svsra,_n_s8,,)(svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_z'}}
  SVE_ACLE_FUNC(svqabs,_s8,_z,)(pg, svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_m'}}
  SVE_ACLE_FUNC(svqabs,_s8,_m,)(svundef_s8(), pg, svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_x'}}
  SVE_ACLE_FUNC(svqabs,_s8,_x,)(pg, svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svcadd_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svcadd'}}
  SVE_ACLE_FUNC(svcadd,_s8,,)(svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_s8,,)(svundef2_s8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'sveortb_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svbcax_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svbcax_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqshlu_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshlu_z'}}
  SVE_ACLE_FUNC(svqshlu,_n_s8,_z,)(pg, svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svcmla_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svcmla'}}
  SVE_ACLE_FUNC(svcmla,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrshr_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshr_z'}}
  SVE_ACLE_FUNC(svrshr,_n_s8,_z,)(pg, svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svaddp_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svtbx_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_s8,,)(svundef_s8(), svundef_s8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqrdcmlah_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdcmlah'}}
  SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svminp_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svminp_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrsra_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsra'}}
  SVE_ACLE_FUNC(svrsra,_n_s8,,)(svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'sveor3_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'sveor3_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh'}}
  SVE_ACLE_FUNC(svqrdmlsh,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh'}}
  SVE_ACLE_FUNC(svqrdmlsh,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmatch_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svmatch'}}
  SVE_ACLE_FUNC(svmatch,_s8,,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_s8,,)(const_i8_ptr, const_i8_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svqcadd_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqcadd'}}
  SVE_ACLE_FUNC(svqcadd,_s8,,)(svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_s8,,)(const_i8_ptr, const_i8_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svsli_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svsli'}}
  SVE_ACLE_FUNC(svsli,_n_s8,,)(svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svnmatch_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svnmatch'}}
  SVE_ACLE_FUNC(svnmatch,_s8,,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svaba_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svaba_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_m'}}
  SVE_ACLE_FUNC(svuqadd,_s8,_m,)(pg, svundef_s8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_m'}}
  SVE_ACLE_FUNC(svuqadd,_n_s8,_m,)(pg, svundef_s8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_z'}}
  SVE_ACLE_FUNC(svuqadd,_s8,_z,)(pg, svundef_s8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_z'}}
  SVE_ACLE_FUNC(svuqadd,_n_s8,_z,)(pg, svundef_s8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_x'}}
  SVE_ACLE_FUNC(svuqadd,_s8,_x,)(pg, svundef_s8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_x'}}
  SVE_ACLE_FUNC(svuqadd,_n_s8,_x,)(pg, svundef_s8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svbsl_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah'}}
  SVE_ACLE_FUNC(svqrdmlah,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah'}}
  SVE_ACLE_FUNC(svqrdmlah,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsri_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svsri'}}
  SVE_ACLE_FUNC(svsri,_n_s8,,)(svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_n_s8,,)(svundef_s8(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_n_s8,_x,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_z'}}
  SVE_ACLE_FUNC(svqneg,_s8,_z,)(pg, svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_m'}}
  SVE_ACLE_FUNC(svqneg,_s8,_m,)(svundef_s8(), pg, svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_x'}}
  SVE_ACLE_FUNC(svqneg,_s8,_x,)(pg, svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svxar_n_s8'}}
  // overload-error@+1 {{use of undeclared identifier 'svxar'}}
  SVE_ACLE_FUNC(svxar,_n_s8,,)(svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_s8,_z,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_s8,_m,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_s8,_x,)(pg, svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_n_s8,_z,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_n_s8,_m,)(pg, svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_n_s8,_x,)(pg, svundef_s8(), i8);
}

void test_s16(svbool_t pg, const int16_t *const_i16_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svmullb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmullb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrshrunb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshrunb'}}
  SVE_ACLE_FUNC(svqrshrunb,_n_s16,,)(svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalbt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalbt'}}
  SVE_ACLE_FUNC(svqdmlalbt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalbt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalbt'}}
  SVE_ACLE_FUNC(svqdmlalbt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh'}}
  SVE_ACLE_FUNC(svqrdmulh,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh'}}
  SVE_ACLE_FUNC(svqrdmulh,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_lane_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh_lane'}}
  SVE_ACLE_FUNC(svqrdmulh_lane,_s16,,)(svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_s16,,)(svundef_s16(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_n_s16,,)(svundef_s16(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh'}}
  SVE_ACLE_FUNC(svqdmulh,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh'}}
  SVE_ACLE_FUNC(svqdmulh,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_lane_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh_lane'}}
  SVE_ACLE_FUNC(svqdmulh_lane,_s16,,)(svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqshrunt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshrunt'}}
  SVE_ACLE_FUNC(svqshrunt,_n_s16,,)(svundef_u8(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_s16,,)(svundef_s8(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_n_s16,,)(svundef_s8(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslb'}}
  SVE_ACLE_FUNC(svqdmlslb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslb'}}
  SVE_ACLE_FUNC(svqdmlslb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_s16,,)(svundef_s8(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_n_s16,,)(svundef_s8(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_z'}}
  SVE_ACLE_FUNC(svqabs,_s16,_z,)(pg, svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_m'}}
  SVE_ACLE_FUNC(svqabs,_s16,_m,)(svundef_s16(), pg, svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_x'}}
  SVE_ACLE_FUNC(svqabs,_s16,_x,)(pg, svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddlbt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlbt'}}
  SVE_ACLE_FUNC(svaddlbt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svaddlbt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlbt'}}
  SVE_ACLE_FUNC(svaddlbt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_s16,,)(svundef2_s16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svshrnt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svshrnt'}}
  SVE_ACLE_FUNC(svshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'sveortb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnb'}}
  SVE_ACLE_FUNC(svqxtnb,_s16,,)(svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svshrnb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svshrnb'}}
  SVE_ACLE_FUNC(svshrnb,_n_s16,,)(svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_s16,,)(svundef_s8(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_n_s16,,)(svundef_s8(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svmls_lane_s16'}}
  // overload-error@+1 {{no matching function for call to 'svmls_lane'}}
  SVE_ACLE_FUNC(svmls_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalt'}}
  SVE_ACLE_FUNC(svqdmlalt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalt'}}
  SVE_ACLE_FUNC(svqdmlalt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svbcax_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svbcax_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnt'}}
  SVE_ACLE_FUNC(svqxtnt,_s16,,)(svundef_s8(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalb'}}
  SVE_ACLE_FUNC(svqdmlalb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalb'}}
  SVE_ACLE_FUNC(svqdmlalb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svsublbt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublbt'}}
  SVE_ACLE_FUNC(svsublbt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsublbt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublbt'}}
  SVE_ACLE_FUNC(svsublbt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqshrnt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshrnt'}}
  SVE_ACLE_FUNC(svqshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmullt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullt'}}
  SVE_ACLE_FUNC(svqdmullt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmullt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullt'}}
  SVE_ACLE_FUNC(svqdmullt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsublt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsublt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslbt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslbt'}}
  SVE_ACLE_FUNC(svqdmlslbt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslbt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslbt'}}
  SVE_ACLE_FUNC(svqdmlslbt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_z'}}
  SVE_ACLE_FUNC(svadalp,_s16,_z,)(pg, svundef_s16(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_m'}}
  SVE_ACLE_FUNC(svadalp,_s16,_m,)(pg, svundef_s16(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_x'}}
  SVE_ACLE_FUNC(svadalp,_s16,_x,)(pg, svundef_s16(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmul_lane_s16'}}
  // overload-error@+1 {{no matching function for call to 'svmul_lane'}}
  SVE_ACLE_FUNC(svmul_lane,_s16,,)(svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_s16,,)(svundef_s16(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_n_s16,,)(svundef_s16(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrshrnt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshrnt'}}
  SVE_ACLE_FUNC(svqrshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svaddp_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svtbx_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_s16,,)(svundef_s16(), svundef_s16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrshrnb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshrnb'}}
  SVE_ACLE_FUNC(svqrshrnb,_n_s16,,)(svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svminp_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svminp_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svabalt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svabalt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqshrnb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshrnb'}}
  SVE_ACLE_FUNC(svqshrnb,_n_s16,,)(svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'sveor3_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'sveor3_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqshrunb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshrunb'}}
  SVE_ACLE_FUNC(svqshrunb,_n_s16,,)(svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svmovlb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlb'}}
  SVE_ACLE_FUNC(svmovlb,_s16,,)(svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh'}}
  SVE_ACLE_FUNC(svqrdmlsh,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh'}}
  SVE_ACLE_FUNC(svqrdmlsh,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_lane_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh_lane'}}
  SVE_ACLE_FUNC(svqrdmlsh_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslt'}}
  SVE_ACLE_FUNC(svqdmlslt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslt'}}
  SVE_ACLE_FUNC(svqdmlslt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svmatch_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmatch'}}
  SVE_ACLE_FUNC(svmatch,_s16,,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqxtunb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtunb'}}
  SVE_ACLE_FUNC(svqxtunb,_s16,,)(svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmla_lane_s16'}}
  // overload-error@+1 {{no matching function for call to 'svmla_lane'}}
  SVE_ACLE_FUNC(svmla_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrshrnb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshrnb'}}
  SVE_ACLE_FUNC(svrshrnb,_n_s16,,)(svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_s16,,)(const_i16_ptr, const_i16_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svshllb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svshllb'}}
  SVE_ACLE_FUNC(svshllb,_n_s16,,)(svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_s16,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_n_s16,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_s16,,)(const_i16_ptr, const_i16_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_s16,,)(svundef_s16(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_n_s16,,)(svundef_s16(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svnmatch_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svnmatch'}}
  SVE_ACLE_FUNC(svnmatch,_s16,,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaba_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaba_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_s16,,)(svundef_s8(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_n_s16,,)(svundef_s8(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_m'}}
  SVE_ACLE_FUNC(svuqadd,_s16,_m,)(pg, svundef_s16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_m'}}
  SVE_ACLE_FUNC(svuqadd,_n_s16,_m,)(pg, svundef_s16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_z'}}
  SVE_ACLE_FUNC(svuqadd,_s16,_z,)(pg, svundef_s16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_z'}}
  SVE_ACLE_FUNC(svuqadd,_n_s16,_z,)(pg, svundef_s16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_x'}}
  SVE_ACLE_FUNC(svuqadd,_s16,_x,)(pg, svundef_s16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_x'}}
  SVE_ACLE_FUNC(svuqadd,_n_s16,_x,)(pg, svundef_s16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svbsl_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svbsl_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svshllt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svshllt'}}
  SVE_ACLE_FUNC(svshllt,_n_s16,,)(svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svsubltb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubltb'}}
  SVE_ACLE_FUNC(svsubltb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsubltb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubltb'}}
  SVE_ACLE_FUNC(svsubltb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah'}}
  SVE_ACLE_FUNC(svqrdmlah,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah'}}
  SVE_ACLE_FUNC(svqrdmlah,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_lane_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah_lane'}}
  SVE_ACLE_FUNC(svqrdmlah_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmullb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullb'}}
  SVE_ACLE_FUNC(svqdmullb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqdmullb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullb'}}
  SVE_ACLE_FUNC(svqdmullb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqxtunt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtunt'}}
  SVE_ACLE_FUNC(svqxtunt,_s16,,)(svundef_u8(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrshrunt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshrunt'}}
  SVE_ACLE_FUNC(svqrshrunt,_n_s16,,)(svundef_u8(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svabalb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsublb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_s16,,)(svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsublb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_n_s16,,)(svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_n_s16,,)(svundef_s16(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_n_s16,_x,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_s16,,)(svundef_s16(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_n_s16,,)(svundef_s16(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_s16,,)(svundef_s16(), svundef_s8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_n_s16,,)(svundef_s16(), svundef_s8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_z'}}
  SVE_ACLE_FUNC(svqneg,_s16,_z,)(pg, svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_m'}}
  SVE_ACLE_FUNC(svqneg,_s16,_m,)(svundef_s16(), pg, svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_x'}}
  SVE_ACLE_FUNC(svqneg,_s16,_x,)(pg, svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmovlt_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlt'}}
  SVE_ACLE_FUNC(svmovlt,_s16,,)(svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrshrnt_n_s16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshrnt'}}
  SVE_ACLE_FUNC(svrshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_s16,_z,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_s16,_m,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_s16,_x,)(pg, svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_n_s16,_z,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_n_s16,_m,)(pg, svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_n_s16,_x,)(pg, svundef_s16(), i16);
}

void test_s32(svbool_t pg, const uint16_t *const_u16_ptr, const int16_t *const_i16_ptr, const int32_t *const_i32_ptr, const int8_t *const_i8_ptr, const uint8_t *const_u8_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svmullb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmullb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svmullb_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb_lane'}}
  SVE_ACLE_FUNC(svmullb_lane,_s32,,)(svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalbt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalbt'}}
  SVE_ACLE_FUNC(svqdmlalbt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalbt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalbt'}}
  SVE_ACLE_FUNC(svqdmlalbt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh'}}
  SVE_ACLE_FUNC(svqrdmulh,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh'}}
  SVE_ACLE_FUNC(svqrdmulh,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_s32,,)(svundef_s32(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_n_s32,,)(svundef_s32(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh'}}
  SVE_ACLE_FUNC(svqdmulh,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh'}}
  SVE_ACLE_FUNC(svqdmulh,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_s32,,)(svundef_s16(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_n_s32,,)(svundef_s16(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslb'}}
  SVE_ACLE_FUNC(svqdmlslb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslb'}}
  SVE_ACLE_FUNC(svqdmlslb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslb_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslb_lane'}}
  SVE_ACLE_FUNC(svqdmlslb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_s32,,)(svundef_s16(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_n_s32,,)(svundef_s16(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_z'}}
  SVE_ACLE_FUNC(svqabs,_s32,_z,)(pg, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_m'}}
  SVE_ACLE_FUNC(svqabs,_s32,_m,)(svundef_s32(), pg, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_x'}}
  SVE_ACLE_FUNC(svqabs,_s32,_x,)(pg, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b8_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b8'}}
  SVE_ACLE_FUNC(svwhilegt_b8,_s32,,)(i32, i32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b16_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b16'}}
  SVE_ACLE_FUNC(svwhilegt_b16,_s32,,)(i32, i32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b32_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b32'}}
  SVE_ACLE_FUNC(svwhilegt_b32,_s32,,)(i32, i32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b64_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b64'}}
  SVE_ACLE_FUNC(svwhilegt_b64,_s32,,)(i32, i32);
  // expected-error@+2 {{use of undeclared identifier 'svaddlbt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlbt'}}
  SVE_ACLE_FUNC(svaddlbt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddlbt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlbt'}}
  SVE_ACLE_FUNC(svaddlbt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_s32,,)(svundef2_s32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhistcnt_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhistcnt_z'}}
  SVE_ACLE_FUNC(svhistcnt,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnb'}}
  SVE_ACLE_FUNC(svqxtnb,_s32,,)(svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt_lane'}}
  SVE_ACLE_FUNC(svmlalt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_s32,,)(svundef_s16(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_n_s32,,)(svundef_s16(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u32base_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_s32'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u32offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u32, offset_s32, )(pg, const_u16_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u32base_offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u32base_index_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_index_s32'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _index_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalt'}}
  SVE_ACLE_FUNC(svqdmlalt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalt'}}
  SVE_ACLE_FUNC(svqdmlalt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalt_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalt_lane'}}
  SVE_ACLE_FUNC(svqdmlalt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svbcax_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svbcax_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnt'}}
  SVE_ACLE_FUNC(svqxtnt,_s32,,)(svundef_s16(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalb'}}
  SVE_ACLE_FUNC(svqdmlalb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalb'}}
  SVE_ACLE_FUNC(svqdmlalb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalb_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalb_lane'}}
  SVE_ACLE_FUNC(svqdmlalb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svcdot_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svcdot'}}
  SVE_ACLE_FUNC(svcdot,_s32,,)(svundef_s32(), svundef_s8(), svundef_s8(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svsublbt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublbt'}}
  SVE_ACLE_FUNC(svsublbt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsublbt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublbt'}}
  SVE_ACLE_FUNC(svsublbt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmullt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullt'}}
  SVE_ACLE_FUNC(svqdmullt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmullt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullt'}}
  SVE_ACLE_FUNC(svqdmullt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmullt_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullt_lane'}}
  SVE_ACLE_FUNC(svqdmullt_lane,_s32,,)(svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svsublt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsublt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslbt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslbt'}}
  SVE_ACLE_FUNC(svqdmlslbt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslbt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslbt'}}
  SVE_ACLE_FUNC(svqdmlslbt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_z'}}
  SVE_ACLE_FUNC(svadalp,_s32,_z,)(pg, svundef_s32(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_m'}}
  SVE_ACLE_FUNC(svadalp,_s32,_m,)(pg, svundef_s32(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_x'}}
  SVE_ACLE_FUNC(svadalp,_s32,_x,)(pg, svundef_s32(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b8_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b8'}}
  SVE_ACLE_FUNC(svwhilege_b8,_s32,,)(i32, i32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b16_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b16'}}
  SVE_ACLE_FUNC(svwhilege_b16,_s32,,)(i32, i32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b32_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b32'}}
  SVE_ACLE_FUNC(svwhilege_b32,_s32,,)(i32, i32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b64_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b64'}}
  SVE_ACLE_FUNC(svwhilege_b64,_s32,,)(i32, i32);
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_s32,,)(svundef_s32(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_n_s32,,)(svundef_s32(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svaddp_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svtbx_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_s32,,)(svundef_s32(), svundef_s32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svminp_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svminp_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svabalt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svabalt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'sveor3_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'sveor3_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svmovlb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlb'}}
  SVE_ACLE_FUNC(svmovlb,_s32,,)(svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, , _s32)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u32, offset, _s32)(pg, i32_ptr, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _offset, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_index_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _index, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh'}}
  SVE_ACLE_FUNC(svqrdmlsh,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh'}}
  SVE_ACLE_FUNC(svqrdmlsh,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslt'}}
  SVE_ACLE_FUNC(svqdmlslt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslt'}}
  SVE_ACLE_FUNC(svqdmlslt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslt_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslt_lane'}}
  SVE_ACLE_FUNC(svqdmlslt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svmullt_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt_lane'}}
  SVE_ACLE_FUNC(svmullt_lane,_s32,,)(svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u32base_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_s32'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u32offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u32, offset_s32, )(pg, const_i16_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u32base_offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u32base_index_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_index_s32'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _index_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqxtunb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtunb'}}
  SVE_ACLE_FUNC(svqxtunb,_s32,,)(svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_s32,,)(const_i32_ptr, const_i32_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_s32,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_n_s32,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_s32,,)(const_i32_ptr, const_i32_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb_lane'}}
  SVE_ACLE_FUNC(svmlalb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u32base_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_s32'}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u32offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, u32, offset_s32, )(pg, const_i8_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u32base_offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_s32,,)(svundef_s32(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_n_s32,,)(svundef_s32(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u32base_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_s32'}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u32offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, u32, offset_s32, )(pg, const_u8_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u32base_offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaba_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaba_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_s32,,)(svundef_s16(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_n_s32,,)(svundef_s16(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_m'}}
  SVE_ACLE_FUNC(svuqadd,_s32,_m,)(pg, svundef_s32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_m'}}
  SVE_ACLE_FUNC(svuqadd,_n_s32,_m,)(pg, svundef_s32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_z'}}
  SVE_ACLE_FUNC(svuqadd,_s32,_z,)(pg, svundef_s32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_z'}}
  SVE_ACLE_FUNC(svuqadd,_n_s32,_z,)(pg, svundef_s32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_x'}}
  SVE_ACLE_FUNC(svuqadd,_s32,_x,)(pg, svundef_s32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_x'}}
  SVE_ACLE_FUNC(svuqadd,_n_s32,_x,)(pg, svundef_s32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svbsl_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svbsl_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svsubltb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubltb'}}
  SVE_ACLE_FUNC(svsubltb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsubltb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubltb'}}
  SVE_ACLE_FUNC(svsubltb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_s32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _s32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _s32)(pg, const_i32_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset_s32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_index_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index_s32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_s32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah'}}
  SVE_ACLE_FUNC(svqrdmlah,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah'}}
  SVE_ACLE_FUNC(svqrdmlah,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqdmullb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullb'}}
  SVE_ACLE_FUNC(svqdmullb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqdmullb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullb'}}
  SVE_ACLE_FUNC(svqdmullb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqdmullb_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullb_lane'}}
  SVE_ACLE_FUNC(svqdmullb_lane,_s32,,)(svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u32base_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, , _s32)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u32offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u32, offset, _s32)(pg, i16_ptr, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u32base_offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, _offset, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u32base_index_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, _index, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u32base_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter'}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u32base, , _s32)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u32offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, u32, offset, _s32)(pg, i8_ptr, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u32base_offset_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u32base, _offset, _s32)(pg, svundef_u32(), i64, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqxtunt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtunt'}}
  SVE_ACLE_FUNC(svqxtunt,_s32,,)(svundef_u16(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svsublb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_s32,,)(svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsublb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_n_s32,,)(svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_n_s32,,)(svundef_s32(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_n_s32,_x,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_s32,,)(svundef_s32(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_n_s32,,)(svundef_s32(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb_lane'}}
  SVE_ACLE_FUNC(svmlslb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_n_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_n_s32,,)(svundef_s32(), svundef_s16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_lane_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt_lane'}}
  SVE_ACLE_FUNC(svmlslt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_z'}}
  SVE_ACLE_FUNC(svqneg,_s32,_z,)(pg, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_m'}}
  SVE_ACLE_FUNC(svqneg,_s32,_m,)(svundef_s32(), pg, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_x'}}
  SVE_ACLE_FUNC(svqneg,_s32,_x,)(pg, svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmovlt_s32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlt'}}
  SVE_ACLE_FUNC(svmovlt,_s32,,)(svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_s32,_z,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_s32,_m,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_s32,_x,)(pg, svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_n_s32,_z,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_n_s32,_m,)(pg, svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_n_s32,_x,)(pg, svundef_s32(), i32);
}

void test_s64(svbool_t pg, const uint16_t *const_u16_ptr, const int16_t *const_i16_ptr, const int64_t *const_i64_ptr, const int8_t *const_i8_ptr, const uint8_t *const_u8_ptr, const int32_t *const_i32_ptr, const uint32_t *const_u32_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svmullb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmullb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalbt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalbt'}}
  SVE_ACLE_FUNC(svqdmlalbt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalbt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalbt'}}
  SVE_ACLE_FUNC(svqdmlalbt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh'}}
  SVE_ACLE_FUNC(svqrdmulh,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmulh_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmulh'}}
  SVE_ACLE_FUNC(svqrdmulh,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_s64,,)(svundef_s64(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_n_s64,,)(svundef_s64(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh'}}
  SVE_ACLE_FUNC(svqdmulh,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqdmulh_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmulh'}}
  SVE_ACLE_FUNC(svqdmulh,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_s64,,)(svundef_s32(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_n_s64,,)(svundef_s32(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslb'}}
  SVE_ACLE_FUNC(svqdmlslb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslb'}}
  SVE_ACLE_FUNC(svqdmlslb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_s64,,)(svundef_s32(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_n_s64,,)(svundef_s32(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_z'}}
  SVE_ACLE_FUNC(svqabs,_s64,_z,)(pg, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_m'}}
  SVE_ACLE_FUNC(svqabs,_s64,_m,)(svundef_s64(), pg, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqabs_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqabs_x'}}
  SVE_ACLE_FUNC(svqabs,_s64,_x,)(pg, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b8_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b8'}}
  SVE_ACLE_FUNC(svwhilegt_b8,_s64,,)(i64, i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b16_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b16'}}
  SVE_ACLE_FUNC(svwhilegt_b16,_s64,,)(i64, i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b32_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b32'}}
  SVE_ACLE_FUNC(svwhilegt_b32,_s64,,)(i64, i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b64_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b64'}}
  SVE_ACLE_FUNC(svwhilegt_b64,_s64,,)(i64, i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddlbt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlbt'}}
  SVE_ACLE_FUNC(svaddlbt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaddlbt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlbt'}}
  SVE_ACLE_FUNC(svaddlbt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_s64,,)(svundef2_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svhistcnt_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhistcnt_z'}}
  SVE_ACLE_FUNC(svhistcnt,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnb'}}
  SVE_ACLE_FUNC(svqxtnb,_s64,,)(svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_s64,,)(svundef_s32(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_n_s64,,)(svundef_s32(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_s64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, s64, offset_s64, )(pg, const_u16_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u64, offset_s64, )(pg, const_u16_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_s64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, s64, index_s64, )(pg, const_u16_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u64, index_s64, )(pg, const_u16_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64base_index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalt'}}
  SVE_ACLE_FUNC(svqdmlalt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalt'}}
  SVE_ACLE_FUNC(svqdmlalt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svbcax_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svbcax_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnt'}}
  SVE_ACLE_FUNC(svqxtnt,_s64,,)(svundef_s32(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalb'}}
  SVE_ACLE_FUNC(svqdmlalb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlalb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlalb'}}
  SVE_ACLE_FUNC(svqdmlalb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsublbt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublbt'}}
  SVE_ACLE_FUNC(svsublbt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsublbt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublbt'}}
  SVE_ACLE_FUNC(svsublbt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqdmullt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullt'}}
  SVE_ACLE_FUNC(svqdmullt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmullt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullt'}}
  SVE_ACLE_FUNC(svqdmullt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svsublt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsublt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslbt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslbt'}}
  SVE_ACLE_FUNC(svqdmlslbt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslbt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslbt'}}
  SVE_ACLE_FUNC(svqdmlslbt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_z'}}
  SVE_ACLE_FUNC(svadalp,_s64,_z,)(pg, svundef_s64(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_m'}}
  SVE_ACLE_FUNC(svadalp,_s64,_m,)(pg, svundef_s64(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_x'}}
  SVE_ACLE_FUNC(svadalp,_s64,_x,)(pg, svundef_s64(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b8_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b8'}}
  SVE_ACLE_FUNC(svwhilege_b8,_s64,,)(i64, i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b16_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b16'}}
  SVE_ACLE_FUNC(svwhilege_b16,_s64,,)(i64, i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b32_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b32'}}
  SVE_ACLE_FUNC(svwhilege_b32,_s64,,)(i64, i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b64_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b64'}}
  SVE_ACLE_FUNC(svwhilege_b64,_s64,,)(i64, i64);
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_s64,,)(svundef_s64(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_n_s64,,)(svundef_s64(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddp_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svtbx_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_s64,,)(svundef_s64(), svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svminp_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svminp_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svabalt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svabalt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'sveor3_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'sveor3_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svmovlb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlb'}}
  SVE_ACLE_FUNC(svmovlb,_s64,,)(svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, , _s64)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, offset, _s64)(pg, i64_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, offset, _s64)(pg, i64_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _offset, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_s64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, index, _s64)(pg, i64_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, index, _s64)(pg, i64_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _index, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh'}}
  SVE_ACLE_FUNC(svqrdmlsh,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlsh_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlsh'}}
  SVE_ACLE_FUNC(svqrdmlsh,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslt'}}
  SVE_ACLE_FUNC(svqdmlslt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmlslt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmlslt'}}
  SVE_ACLE_FUNC(svqdmlslt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_s64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, s64, offset_s64, )(pg, const_i16_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u64, offset_s64, )(pg, const_i16_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_s64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, s64, index_s64, )(pg, const_i16_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u64, index_s64, )(pg, const_i16_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64base_index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqxtunb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtunb'}}
  SVE_ACLE_FUNC(svqxtunb,_s64,,)(svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_s64,,)(const_i64_ptr, const_i64_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_s64,,)(svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_n_s64,,)(svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_s64,,)(const_i64_ptr, const_i64_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_s64'}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, s64, offset_s64, )(pg, const_i8_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, u64, offset_s64, )(pg, const_i8_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_s64,,)(svundef_s64(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_n_s64,,)(svundef_s64(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_s64'}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, s64, offset_s64, )(pg, const_u8_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, u64, offset_s64, )(pg, const_u8_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaba_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svaba_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_s64,,)(svundef_s32(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_n_s64,,)(svundef_s32(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_m'}}
  SVE_ACLE_FUNC(svuqadd,_s64,_m,)(pg, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_m'}}
  SVE_ACLE_FUNC(svuqadd,_n_s64,_m,)(pg, svundef_s64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_z'}}
  SVE_ACLE_FUNC(svuqadd,_s64,_z,)(pg, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_z'}}
  SVE_ACLE_FUNC(svuqadd,_n_s64,_z,)(pg, svundef_s64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_x'}}
  SVE_ACLE_FUNC(svuqadd,_s64,_x,)(pg, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svuqadd_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svuqadd_x'}}
  SVE_ACLE_FUNC(svuqadd,_n_s64,_x,)(pg, svundef_s64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_s64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, s64, offset_s64, )(pg, const_i32_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, u64, offset_s64, )(pg, const_i32_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_s64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, s64, index_s64, )(pg, const_i32_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, u64, index_s64, )(pg, const_i32_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64base_index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svbsl_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svbsl_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsubltb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubltb'}}
  SVE_ACLE_FUNC(svsubltb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsubltb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubltb'}}
  SVE_ACLE_FUNC(svsubltb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_s64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _s64)(pg, const_i64_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _s64)(pg, const_i64_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_s64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index'}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _s64)(pg, const_i64_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _s64)(pg, const_i64_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah'}}
  SVE_ACLE_FUNC(svqrdmlah,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrdmlah_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrdmlah'}}
  SVE_ACLE_FUNC(svqrdmlah,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqdmullb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullb'}}
  SVE_ACLE_FUNC(svqdmullb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqdmullb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqdmullb'}}
  SVE_ACLE_FUNC(svqdmullb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_s64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _s64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, s64, offset_s64, )(pg, const_u32_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, u64, offset_s64, )(pg, const_u32_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_offset_s64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _offset_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_s64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, s64, index_s64, )(pg, const_u32_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, u64, index_s64, )(pg, const_u32_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64base_index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_index_s64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _index_s64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, , _s64)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, s64, offset, _s64)(pg, i16_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u64, offset, _s64)(pg, i16_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, _offset, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_s64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, s64, index, _s64)(pg, i16_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u64, index, _s64)(pg, i16_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64base_index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, _index, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter'}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u64base, , _s64)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, s64, offset, _s64)(pg, i8_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, u64, offset, _s64)(pg, i8_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u64base, _offset, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64base_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter'}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, , _s64)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_s64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, s64, offset, _s64)(pg, i32_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, u64, offset, _s64)(pg, i32_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64base_offset_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, _offset, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_s64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, s64, index, _s64)(pg, i32_ptr, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, u64, index, _s64)(pg, i32_ptr, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64base_index_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, _index, _s64)(pg, svundef_u64(), i64, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqxtunt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtunt'}}
  SVE_ACLE_FUNC(svqxtunt,_s64,,)(svundef_u32(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svsublb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_s64,,)(svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsublb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_n_s64,,)(svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_n_s64,,)(svundef_s64(), svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_n_s64,_x,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_s64,,)(svundef_s64(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_n_s64,,)(svundef_s64(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_n_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_n_s64,,)(svundef_s64(), svundef_s32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_z'}}
  SVE_ACLE_FUNC(svqneg,_s64,_z,)(pg, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_m'}}
  SVE_ACLE_FUNC(svqneg,_s64,_m,)(svundef_s64(), pg, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqneg_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqneg_x'}}
  SVE_ACLE_FUNC(svqneg,_s64,_x,)(pg, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svmovlt_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlt'}}
  SVE_ACLE_FUNC(svmovlt,_s64,,)(svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_s64,_z,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_s64,_m,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_s64,_x,)(pg, svundef_s64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_n_s64,_z,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_n_s64,_m,)(pg, svundef_s64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_s64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_n_s64,_x,)(pg, svundef_s64(), i64);
}

void test_u8(svbool_t pg, const uint8_t *const_u8_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svhistseg_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svhistseg'}}
  SVE_ACLE_FUNC(svhistseg,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_pair_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullb_pair'}}
  SVE_ACLE_FUNC(svpmullb_pair,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_pair_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullb_pair'}}
  SVE_ACLE_FUNC(svpmullb_pair,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_u8,,)(svundef2_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svpmul_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmul'}}
  SVE_ACLE_FUNC(svpmul,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svpmul_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmul'}}
  SVE_ACLE_FUNC(svpmul,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'sveortb_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svbcax_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbcax_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_u8,_z,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_u8,_m,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_u8,_x,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_n_u8,_z,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_n_u8,_m,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_n_u8,_x,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_pair_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullt_pair'}}
  SVE_ACLE_FUNC(svpmullt_pair,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_pair_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullt_pair'}}
  SVE_ACLE_FUNC(svpmullt_pair,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svaddp_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svtbx_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svminp_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svminp_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_m'}}
  SVE_ACLE_FUNC(svsqadd,_u8,_m,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_m'}}
  SVE_ACLE_FUNC(svsqadd,_n_u8,_m,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_z'}}
  SVE_ACLE_FUNC(svsqadd,_u8,_z,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_z'}}
  SVE_ACLE_FUNC(svsqadd,_n_u8,_z,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_x'}}
  SVE_ACLE_FUNC(svsqadd,_u8,_x,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_x'}}
  SVE_ACLE_FUNC(svsqadd,_n_u8,_x,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'sveor3_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'sveor3_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmatch_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svmatch'}}
  SVE_ACLE_FUNC(svmatch,_u8,,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_u8,,)(const_u8_ptr, const_u8_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_u8,,)(const_u8_ptr, const_u8_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svnmatch_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svnmatch'}}
  SVE_ACLE_FUNC(svnmatch,_u8,,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaba_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaba_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbsl_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_u8,_z,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_u8,_m,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_u8,_x,)(pg, svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_n_u8,_z,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_n_u8,_m,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_n_u8,_x,)(pg, svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_n_u8,,)(svundef_u8(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_u8,_z,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_u8,_m,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_u8,_x,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_n_u8,_z,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_n_u8,_m,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_n_u8,_x,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_u8,_z,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_u8,_m,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_u8,_x,)(pg, svundef_u8(), svundef_s8());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u8_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_n_u8,_z,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u8_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_n_u8,_m,)(pg, svundef_u8(), i8);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u8_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_n_u8,_x,)(pg, svundef_u8(), i8);
}

void test_u16(svbool_t pg, const uint16_t *const_u16_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svmullb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmullb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullb'}}
  SVE_ACLE_FUNC(svpmullb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullb'}}
  SVE_ACLE_FUNC(svpmullb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_u16,,)(svundef_u16(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_n_u16,,)(svundef_u16(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_u16,,)(svundef_u8(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_n_u16,,)(svundef_u8(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_u16,,)(svundef_u8(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_n_u16,,)(svundef_u8(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_u16,,)(svundef2_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'sveortb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnb'}}
  SVE_ACLE_FUNC(svqxtnb,_u16,,)(svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_u16,,)(svundef_u8(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_n_u16,,)(svundef_u8(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svbcax_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svbcax_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnt'}}
  SVE_ACLE_FUNC(svqxtnt,_u16,,)(svundef_u8(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_u16,_z,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_u16,_m,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_u16,_x,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_n_u16,_z,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_n_u16,_m,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_n_u16,_x,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svsublt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svsublt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_z'}}
  SVE_ACLE_FUNC(svadalp,_u16,_z,)(pg, svundef_u16(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_m'}}
  SVE_ACLE_FUNC(svadalp,_u16,_m,)(pg, svundef_u16(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_x'}}
  SVE_ACLE_FUNC(svadalp,_u16,_x,)(pg, svundef_u16(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullt'}}
  SVE_ACLE_FUNC(svpmullt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullt'}}
  SVE_ACLE_FUNC(svpmullt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_u16,,)(svundef_u16(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_n_u16,,)(svundef_u16(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svaddp_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svtbx_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svminp_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svminp_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_m'}}
  SVE_ACLE_FUNC(svsqadd,_u16,_m,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_m'}}
  SVE_ACLE_FUNC(svsqadd,_n_u16,_m,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_z'}}
  SVE_ACLE_FUNC(svsqadd,_u16,_z,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_z'}}
  SVE_ACLE_FUNC(svsqadd,_n_u16,_z,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_x'}}
  SVE_ACLE_FUNC(svsqadd,_u16,_x,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_x'}}
  SVE_ACLE_FUNC(svsqadd,_n_u16,_x,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svabalt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svabalt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'sveor3_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'sveor3_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svmovlb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlb'}}
  SVE_ACLE_FUNC(svmovlb,_u16,,)(svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svmatch_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmatch'}}
  SVE_ACLE_FUNC(svmatch,_u16,,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_u16,,)(const_u16_ptr, const_u16_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_u16,,)(const_u16_ptr, const_u16_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_u16,,)(svundef_u16(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_n_u16,,)(svundef_u16(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svnmatch_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svnmatch'}}
  SVE_ACLE_FUNC(svnmatch,_u16,,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaba_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaba_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_u16,,)(svundef_u8(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_n_u16,,)(svundef_u8(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svbsl_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svbsl_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_u16,_z,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_u16,_m,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_u16,_x,)(pg, svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_n_u16,_z,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_n_u16,_m,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_n_u16,_x,)(pg, svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svabalb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svsublb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_u16,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svsublb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_n_u16,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_n_u16,,)(svundef_u16(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_u16,_z,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_u16,_m,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_u16,_x,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_n_u16,_z,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_n_u16,_m,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_n_u16,_x,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_u16,,)(svundef_u16(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_n_u16,,)(svundef_u16(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_u16,,)(svundef_u16(), svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_n_u16,,)(svundef_u16(), svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svmovlt_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlt'}}
  SVE_ACLE_FUNC(svmovlt,_u16,,)(svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_u16,_z,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_u16,_m,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_u16,_x,)(pg, svundef_u16(), svundef_s16());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_n_u16,_z,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_n_u16,_m,)(pg, svundef_u16(), i16);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_n_u16,_x,)(pg, svundef_u16(), i16);
}

void test_u32(svbool_t pg, const uint16_t *const_u16_ptr, const int16_t *const_i16_ptr, const uint32_t *const_u32_ptr, const int8_t *const_i8_ptr, const uint8_t *const_u8_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svmullb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmullb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_pair_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullb_pair'}}
  SVE_ACLE_FUNC(svpmullb_pair,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_pair_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullb_pair'}}
  SVE_ACLE_FUNC(svpmullb_pair,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_u32,,)(svundef_u32(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_n_u32,,)(svundef_u32(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_u32,,)(svundef_u16(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_n_u32,,)(svundef_u16(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_u32,,)(svundef_u16(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_n_u32,,)(svundef_u16(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b8_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b8'}}
  SVE_ACLE_FUNC(svwhilegt_b8,_u32,,)(u32, u32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b16_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b16'}}
  SVE_ACLE_FUNC(svwhilegt_b16,_u32,,)(u32, u32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b32_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b32'}}
  SVE_ACLE_FUNC(svwhilegt_b32,_u32,,)(u32, u32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b64_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b64'}}
  SVE_ACLE_FUNC(svwhilegt_b64,_u32,,)(u32, u32);
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_u32,,)(svundef2_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhistcnt_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhistcnt_z'}}
  SVE_ACLE_FUNC(svhistcnt,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnb'}}
  SVE_ACLE_FUNC(svqxtnb,_u32,,)(svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_u32,,)(svundef_u16(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_n_u32,,)(svundef_u16(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u32base_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_u32'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u32offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u32, offset_u32, )(pg, const_u16_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u32base_offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u32base_index_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_index_u32'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u32base, _index_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svbcax_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svbcax_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnt'}}
  SVE_ACLE_FUNC(svqxtnt,_u32,,)(svundef_u16(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_u32,_z,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_u32,_m,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_u32,_x,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_n_u32,_z,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_n_u32,_m,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_n_u32,_x,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svsublt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svsublt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_z'}}
  SVE_ACLE_FUNC(svadalp,_u32,_z,)(pg, svundef_u32(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_m'}}
  SVE_ACLE_FUNC(svadalp,_u32,_m,)(pg, svundef_u32(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_x'}}
  SVE_ACLE_FUNC(svadalp,_u32,_x,)(pg, svundef_u32(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b8_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b8'}}
  SVE_ACLE_FUNC(svwhilege_b8,_u32,,)(u32, u32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b16_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b16'}}
  SVE_ACLE_FUNC(svwhilege_b16,_u32,,)(u32, u32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b32_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b32'}}
  SVE_ACLE_FUNC(svwhilege_b32,_u32,,)(u32, u32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b64_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b64'}}
  SVE_ACLE_FUNC(svwhilege_b64,_u32,,)(u32, u32);
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_pair_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullt_pair'}}
  SVE_ACLE_FUNC(svpmullt_pair,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_pair_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullt_pair'}}
  SVE_ACLE_FUNC(svpmullt_pair,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_u32,,)(svundef_u32(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_n_u32,,)(svundef_u32(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svadclt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svadclt'}}
  SVE_ACLE_FUNC(svadclt,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svadclt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svadclt'}}
  SVE_ACLE_FUNC(svadclt,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svaddp_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrecpe_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrecpe_z'}}
  SVE_ACLE_FUNC(svrecpe,_u32,_z,)(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrecpe_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrecpe_m'}}
  SVE_ACLE_FUNC(svrecpe,_u32,_m,)(svundef_u32(), pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrecpe_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrecpe_x'}}
  SVE_ACLE_FUNC(svrecpe,_u32,_x,)(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svtbx_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svminp_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svminp_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_m'}}
  SVE_ACLE_FUNC(svsqadd,_u32,_m,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_m'}}
  SVE_ACLE_FUNC(svsqadd,_n_u32,_m,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_z'}}
  SVE_ACLE_FUNC(svsqadd,_u32,_z,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_z'}}
  SVE_ACLE_FUNC(svsqadd,_n_u32,_z,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_x'}}
  SVE_ACLE_FUNC(svsqadd,_u32,_x,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_x'}}
  SVE_ACLE_FUNC(svsqadd,_n_u32,_x,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svabalt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svabalt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'sveor3_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'sveor3_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svmovlb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlb'}}
  SVE_ACLE_FUNC(svmovlb,_u32,,)(svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, , _u32)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u32, offset, _u32)(pg, u32_ptr, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _offset, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_index_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _index, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsbclt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsbclt'}}
  SVE_ACLE_FUNC(svsbclt,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsbclt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsbclt'}}
  SVE_ACLE_FUNC(svsbclt,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svmullt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u32base_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_u32'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u32offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u32, offset_u32, )(pg, const_i16_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u32base_offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u32base_index_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_index_u32'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u32base, _index_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_u32,,)(const_u32_ptr, const_u32_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_u32,,)(const_u32_ptr, const_u32_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u32base_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_u32'}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u32offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, u32, offset_u32, )(pg, const_i8_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u32base_offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_u32,,)(svundef_u32(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_n_u32,,)(svundef_u32(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u32base_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_u32'}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u32offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, u32, offset_u32, )(pg, const_u8_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u32base_offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaba_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaba_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_u32,,)(svundef_u16(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_n_u32,,)(svundef_u16(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svbsl_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svbsl_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svadclb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svadclb'}}
  SVE_ACLE_FUNC(svadclb,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svadclb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svadclb'}}
  SVE_ACLE_FUNC(svadclb,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_u32,_z,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_u32,_m,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_u32,_x,)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_n_u32,_z,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_n_u32,_m,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_n_u32,_x,)(pg, svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_u32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _u32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _u32)(pg, const_u32_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset_u32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_index_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index_u32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_u32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u32base_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, , _u32)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u32offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u32, offset, _u32)(pg, u16_ptr, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u32base_offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, _offset, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u32base_index_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u32base, _index, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u32base_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter'}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u32base, , _u32)(pg, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u32offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, u32, offset, _u32)(pg, u8_ptr, svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u32base_offset_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u32base, _offset, _u32)(pg, svundef_u32(), i64, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svabalb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svsublb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_u32,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svsublb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_n_u32,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svsbclb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsbclb'}}
  SVE_ACLE_FUNC(svsbclb,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsbclb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsbclb'}}
  SVE_ACLE_FUNC(svsbclb,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_n_u32,,)(svundef_u32(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_u32,_z,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_u32,_m,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_u32,_x,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_n_u32,_z,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_n_u32,_m,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_n_u32,_x,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svrsqrte_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsqrte_z'}}
  SVE_ACLE_FUNC(svrsqrte,_u32,_z,)(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrsqrte_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsqrte_m'}}
  SVE_ACLE_FUNC(svrsqrte,_u32,_m,)(svundef_u32(), pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svrsqrte_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsqrte_x'}}
  SVE_ACLE_FUNC(svrsqrte,_u32,_x,)(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_u32,,)(svundef_u32(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_n_u32,,)(svundef_u32(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_n_u32,,)(svundef_u32(), svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svmovlt_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlt'}}
  SVE_ACLE_FUNC(svmovlt,_u32,,)(svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_u32,_z,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_u32,_m,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_u32,_x,)(pg, svundef_u32(), svundef_s32());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_n_u32,_z,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_n_u32,_m,)(pg, svundef_u32(), i32);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_n_u32,_x,)(pg, svundef_u32(), i32);
}

void test_u64(svbool_t pg, const uint16_t *const_u16_ptr, const int16_t *const_i16_ptr, const uint64_t *const_u64_ptr, const int8_t *const_i8_ptr, const uint8_t *const_u8_ptr, const int32_t *const_i32_ptr, const uint32_t *const_u32_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svmullb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmullb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullb'}}
  SVE_ACLE_FUNC(svmullb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullb'}}
  SVE_ACLE_FUNC(svpmullb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullb'}}
  SVE_ACLE_FUNC(svpmullb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_u64,,)(svundef_u64(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddwb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwb'}}
  SVE_ACLE_FUNC(svaddwb,_n_u64,,)(svundef_u64(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnb'}}
  SVE_ACLE_FUNC(svsubhnb,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_u64,,)(svundef_u32(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnt'}}
  SVE_ACLE_FUNC(svrsubhnt,_n_u64,,)(svundef_u32(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svnbsl_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svnbsl'}}
  SVE_ACLE_FUNC(svnbsl,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_u64,,)(svundef_u32(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svsubhnt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubhnt'}}
  SVE_ACLE_FUNC(svsubhnt,_n_u64,,)(svundef_u32(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b8_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b8'}}
  SVE_ACLE_FUNC(svwhilegt_b8,_u64,,)(u64, u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b16_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b16'}}
  SVE_ACLE_FUNC(svwhilegt_b16,_u64,,)(u64, u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b32_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b32'}}
  SVE_ACLE_FUNC(svwhilegt_b32,_u64,,)(u64, u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilegt_b64_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilegt_b64'}}
  SVE_ACLE_FUNC(svwhilegt_b64,_u64,,)(u64, u64);
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_u64,,)(svundef2_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_z'}}
  SVE_ACLE_FUNC(svhsubr,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_m'}}
  SVE_ACLE_FUNC(svhsubr,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhsubr_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsubr_x'}}
  SVE_ACLE_FUNC(svhsubr,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhistcnt_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhistcnt_z'}}
  SVE_ACLE_FUNC(svhistcnt,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'sveortb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveortb'}}
  SVE_ACLE_FUNC(sveortb,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnb'}}
  SVE_ACLE_FUNC(svqxtnb,_u64,,)(svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmlalt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalt'}}
  SVE_ACLE_FUNC(svmlalt,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_u64,,)(svundef_u32(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnt'}}
  SVE_ACLE_FUNC(svaddhnt,_n_u64,,)(svundef_u32(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_u64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, s64, offset_u64, )(pg, const_u16_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u64, offset_u64, )(pg, const_u16_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_s64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, s64, index_u64, )(pg, const_u16_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather_, u64, index_u64, )(pg, const_u16_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uh_gather_u64base_index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uh_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1uh_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svbcax_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svbcax_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbcax'}}
  SVE_ACLE_FUNC(svbcax,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqxtnt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svqxtnt'}}
  SVE_ACLE_FUNC(svqxtnt,_u64,,)(svundef_u32(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_u64,_z,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_u64,_m,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_u64,_x,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_z'}}
  SVE_ACLE_FUNC(svqrshl,_n_u64,_z,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_m'}}
  SVE_ACLE_FUNC(svqrshl,_n_u64,_m,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqrshl_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqrshl_x'}}
  SVE_ACLE_FUNC(svqrshl,_n_u64,_x,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsublt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsublt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublt'}}
  SVE_ACLE_FUNC(svsublt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_z'}}
  SVE_ACLE_FUNC(svadalp,_u64,_z,)(pg, svundef_u64(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_m'}}
  SVE_ACLE_FUNC(svadalp,_u64,_m,)(pg, svundef_u64(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svadalp_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svadalp_x'}}
  SVE_ACLE_FUNC(svadalp,_u64,_x,)(pg, svundef_u64(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b8_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b8'}}
  SVE_ACLE_FUNC(svwhilege_b8,_u64,,)(u64, u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b16_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b16'}}
  SVE_ACLE_FUNC(svwhilege_b16,_u64,,)(u64, u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b32_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b32'}}
  SVE_ACLE_FUNC(svwhilege_b32,_u64,,)(u64, u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilege_b64_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilege_b64'}}
  SVE_ACLE_FUNC(svwhilege_b64,_u64,,)(u64, u64);
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullt'}}
  SVE_ACLE_FUNC(svpmullt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svpmullt'}}
  SVE_ACLE_FUNC(svpmullt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_u64,,)(svundef_u64(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsubwt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwt'}}
  SVE_ACLE_FUNC(svsubwt,_n_u64,,)(svundef_u64(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_z'}}
  SVE_ACLE_FUNC(svqsubr,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_m'}}
  SVE_ACLE_FUNC(svqsubr,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqsubr_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsubr_x'}}
  SVE_ACLE_FUNC(svqsubr,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svadclt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svadclt'}}
  SVE_ACLE_FUNC(svadclt,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svadclt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svadclt'}}
  SVE_ACLE_FUNC(svadclt,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svaddp_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_m'}}
  SVE_ACLE_FUNC(svqadd,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_z'}}
  SVE_ACLE_FUNC(svqadd,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqadd_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqadd_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqadd_x'}}
  SVE_ACLE_FUNC(svqadd,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svabdlb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlb'}}
  SVE_ACLE_FUNC(svabdlb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svtbx_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svabdlt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabdlt'}}
  SVE_ACLE_FUNC(svabdlt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svminp_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svminp_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_m'}}
  SVE_ACLE_FUNC(svsqadd,_u64,_m,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_m'}}
  SVE_ACLE_FUNC(svsqadd,_n_u64,_m,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_z'}}
  SVE_ACLE_FUNC(svsqadd,_u64,_z,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_z'}}
  SVE_ACLE_FUNC(svsqadd,_n_u64,_z,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_x'}}
  SVE_ACLE_FUNC(svsqadd,_u64,_x,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svsqadd_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svsqadd_x'}}
  SVE_ACLE_FUNC(svsqadd,_n_u64,_x,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_z'}}
  SVE_ACLE_FUNC(svqsub,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_m'}}
  SVE_ACLE_FUNC(svqsub,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svqsub_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqsub_x'}}
  SVE_ACLE_FUNC(svqsub,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svrsubhnb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrsubhnb'}}
  SVE_ACLE_FUNC(svrsubhnb,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svaddhnb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddhnb'}}
  SVE_ACLE_FUNC(svaddhnb,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svabalt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svabalt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalt'}}
  SVE_ACLE_FUNC(svabalt,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'sveor3_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'sveor3_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveor3'}}
  SVE_ACLE_FUNC(sveor3,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_m'}}
  SVE_ACLE_FUNC(svhadd,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_z'}}
  SVE_ACLE_FUNC(svhadd,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhadd_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhadd_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhadd_x'}}
  SVE_ACLE_FUNC(svhadd,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svmovlb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlb'}}
  SVE_ACLE_FUNC(svmovlb,_u64,,)(svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, , _u64)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, offset, _u64)(pg, u64_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, offset, _u64)(pg, u64_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _offset, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_s64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, index, _u64)(pg, u64_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, index, _u64)(pg, u64_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _index, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svsbclt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsbclt'}}
  SVE_ACLE_FUNC(svsbclt,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svsbclt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsbclt'}}
  SVE_ACLE_FUNC(svsbclt,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svmullt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmullt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmullt'}}
  SVE_ACLE_FUNC(svmullt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_u64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, s64, offset_u64, )(pg, const_i16_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u64, offset_u64, )(pg, const_i16_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_s64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, s64, index_u64, )(pg, const_i16_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather_, u64, index_u64, )(pg, const_i16_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sh_gather_u64base_index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sh_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1sh_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_u64,,)(const_u64_ptr, const_u64_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_m'}}
  SVE_ACLE_FUNC(svrhadd,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_z'}}
  SVE_ACLE_FUNC(svrhadd,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svrhadd_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrhadd_x'}}
  SVE_ACLE_FUNC(svrhadd,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnb'}}
  SVE_ACLE_FUNC(svraddhnb,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_u64,,)(const_u64_ptr, const_u64_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmlalb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlalb'}}
  SVE_ACLE_FUNC(svmlalb,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_u64'}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, s64, offset_u64, )(pg, const_i8_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sb_gather_, u64, offset_u64, )(pg, const_i8_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sb_gather_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sb_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sb_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_u64,,)(svundef_u64(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsubwb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsubwb'}}
  SVE_ACLE_FUNC(svsubwb,_n_u64,,)(svundef_u64(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_u64'}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, s64, offset_u64, )(pg, const_u8_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1ub_gather_, u64, offset_u64, )(pg, const_u8_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1ub_gather_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1ub_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1ub_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaba_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svaba_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaba'}}
  SVE_ACLE_FUNC(svaba,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_u64,,)(svundef_u32(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svraddhnt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svraddhnt'}}
  SVE_ACLE_FUNC(svraddhnt,_n_u64,,)(svundef_u32(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'sveorbt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'sveorbt'}}
  SVE_ACLE_FUNC(sveorbt,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_u64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, s64, offset_u64, )(pg, const_i32_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, u64, offset_u64, )(pg, const_i32_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_s64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, s64, index_u64, )(pg, const_i32_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather_, u64, index_u64, )(pg, const_i32_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1sw_gather_u64base_index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1sw_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1sw_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svbsl_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svbsl_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl'}}
  SVE_ACLE_FUNC(svbsl,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svadclb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svadclb'}}
  SVE_ACLE_FUNC(svadclb,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svadclb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svadclb'}}
  SVE_ACLE_FUNC(svadclb,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_u64,_z,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_u64,_m,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_u64,_x,)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_z'}}
  SVE_ACLE_FUNC(svhsub,_n_u64,_z,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_m'}}
  SVE_ACLE_FUNC(svhsub,_n_u64,_m,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svhsub_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svhsub_x'}}
  SVE_ACLE_FUNC(svhsub,_n_u64,_x,)(pg, svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_u64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _u64)(pg, const_u64_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _u64)(pg, const_u64_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_s64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index'}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _u64)(pg, const_u64_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _u64)(pg, const_u64_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddlb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlb'}}
  SVE_ACLE_FUNC(svaddlb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_u64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _u64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, s64, offset_u64, )(pg, const_u32_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, u64, offset_u64, )(pg, const_u32_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_offset_u64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _offset_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_s64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, s64, index_u64, )(pg, const_u32_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather_, u64, index_u64, )(pg, const_u32_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1uw_gather_u64base_index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1uw_gather_index_u64'}}
  SVE_ACLE_FUNC(svldnt1uw_gather, _u64base, _index_u64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, , _u64)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, s64, offset, _u64)(pg, u16_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u64, offset, _u64)(pg, u16_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, _offset, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_s64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, s64, index, _u64)(pg, u16_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1h_scatter_, u64, index, _u64)(pg, u16_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1h_scatter_u64base_index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1h_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1h_scatter, _u64base, _index, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter'}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u64base, , _u64)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, s64, offset, _u64)(pg, u8_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter_, u64, offset, _u64)(pg, u8_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1b_scatter_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1b_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1b_scatter, _u64base, _offset, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svbsl2n_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl2n'}}
  SVE_ACLE_FUNC(svbsl2n,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddlt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddlt'}}
  SVE_ACLE_FUNC(svaddlt,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64base_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter'}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, , _u64)(pg, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_s64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, s64, offset, _u64)(pg, u32_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, u64, offset, _u64)(pg, u32_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64base_offset_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, _offset, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_s64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, s64, index, _u64)(pg, u32_ptr, svundef_s64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1w_scatter_, u64, index, _u64)(pg, u32_ptr, svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1w_scatter_u64base_index_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1w_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1w_scatter, _u64base, _index, _u64)(pg, svundef_u64(), i64, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svabalb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svabalb'}}
  SVE_ACLE_FUNC(svabalb,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svsublb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_u64,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsublb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsublb'}}
  SVE_ACLE_FUNC(svsublb,_n_u64,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svsbclb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsbclb'}}
  SVE_ACLE_FUNC(svsbclb,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svsbclb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svsbclb'}}
  SVE_ACLE_FUNC(svsbclb,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svbsl1n_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbsl1n'}}
  SVE_ACLE_FUNC(svbsl1n,_n_u64,,)(svundef_u64(), svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_u64,_z,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_u64,_m,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_u64,_x,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_z'}}
  SVE_ACLE_FUNC(svrshl,_n_u64,_z,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_m'}}
  SVE_ACLE_FUNC(svrshl,_n_u64,_m,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svrshl_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svrshl_x'}}
  SVE_ACLE_FUNC(svrshl,_n_u64,_x,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_u64,,)(svundef_u64(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddwt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddwt'}}
  SVE_ACLE_FUNC(svaddwt,_n_u64,,)(svundef_u64(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmlslb_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslb'}}
  SVE_ACLE_FUNC(svmlslb,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svmlslt_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmlslt'}}
  SVE_ACLE_FUNC(svmlslt,_n_u64,,)(svundef_u64(), svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svmovlt_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svmovlt'}}
  SVE_ACLE_FUNC(svmovlt,_u64,,)(svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_u64,_z,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_u64,_m,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_u64,_x,)(pg, svundef_u64(), svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_z'}}
  SVE_ACLE_FUNC(svqshl,_n_u64,_z,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_m'}}
  SVE_ACLE_FUNC(svqshl,_n_u64,_m,)(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svqshl_n_u64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svqshl_x'}}
  SVE_ACLE_FUNC(svqshl,_n_u64,_x,)(pg, svundef_u64(), i64);
}

void test_f16(svbool_t pg, const float16_t *const_f16_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f16_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_z'}}
  SVE_ACLE_FUNC(svlogb,_f16,_z,)(pg, svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_m'}}
  SVE_ACLE_FUNC(svlogb,_f16,_m,)(svundef_s16(), pg, svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_x'}}
  SVE_ACLE_FUNC(svlogb,_f16,_x,)(pg, svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svminnmp_f16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminnmp_m'}}
  SVE_ACLE_FUNC(svminnmp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svminnmp_f16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminnmp_x'}}
  SVE_ACLE_FUNC(svminnmp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_f16'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_f16,,)(svundef2_f16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_f16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_f16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svtbx_f16'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_f16,,)(svundef_f16(), svundef_f16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svminp_f16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svminp_f16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_f16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_f16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svmaxnmp_f16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxnmp_m'}}
  SVE_ACLE_FUNC(svmaxnmp,_f16,_m,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svmaxnmp_f16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxnmp_x'}}
  SVE_ACLE_FUNC(svmaxnmp,_f16,_x,)(pg, svundef_f16(), svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_f16'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_f16,,)(const_f16_ptr, const_f16_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_f16'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_f16,,)(const_f16_ptr, const_f16_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svcvtlt_f32_f16_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtlt_f32_m'}}
  SVE_ACLE_FUNC(svcvtlt_f32,_f16,_m,)(svundef_f32(), pg, svundef_f16());
  // expected-error@+2 {{use of undeclared identifier 'svcvtlt_f32_f16_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtlt_f32_x'}}
  SVE_ACLE_FUNC(svcvtlt_f32,_f16,_x,)(pg, svundef_f16());
}

void test_f32(svbool_t pg, const float32_t *const_f32_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f32_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_z'}}
  SVE_ACLE_FUNC(svlogb,_f32,_z,)(pg, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_m'}}
  SVE_ACLE_FUNC(svlogb,_f32,_m,)(svundef_s32(), pg, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_x'}}
  SVE_ACLE_FUNC(svlogb,_f32,_x,)(pg, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svminnmp_f32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminnmp_m'}}
  SVE_ACLE_FUNC(svminnmp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svminnmp_f32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminnmp_x'}}
  SVE_ACLE_FUNC(svminnmp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_f32,,)(svundef2_f32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_f32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_f32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svtbx_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_f32,,)(svundef_f32(), svundef_f32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svminp_f32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svminp_f32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, , _f32)(pg, svundef_u32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32offset_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u32, offset, _f32)(pg, f32_ptr, svundef_u32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_offset_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _offset, _f32)(pg, svundef_u32(), i64, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u32base_index_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u32base, _index, _f32)(pg, svundef_u32(), i64, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_f32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_f32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svmaxnmp_f32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxnmp_m'}}
  SVE_ACLE_FUNC(svmaxnmp,_f32,_m,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svmaxnmp_f32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxnmp_x'}}
  SVE_ACLE_FUNC(svmaxnmp,_f32,_x,)(pg, svundef_f32(), svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_f32,,)(const_f32_ptr, const_f32_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svcvtnt_f16_f32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtnt_f16_m'}}
  SVE_ACLE_FUNC(svcvtnt_f16,_f32,_m,)(svundef_f16(), pg, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svcvtnt_f16_f32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtnt_f16_x'}}
  SVE_ACLE_FUNC(svcvtnt_f16,_f32,_x,)(svundef_f16(), pg, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_f32,,)(const_f32_ptr, const_f32_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svcvtlt_f64_f32_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtlt_f64_m'}}
  SVE_ACLE_FUNC(svcvtlt_f64,_f32,_m,)(svundef_f64(), pg, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svcvtlt_f64_f32_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtlt_f64_x'}}
  SVE_ACLE_FUNC(svcvtlt_f64,_f32,_x,)(pg, svundef_f32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_f32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _f32, )(pg, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32offset_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _f32)(pg, const_f32_ptr, svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_offset_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset_f32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_f32, )(pg, svundef_u32(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u32base_index_f32'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index_f32'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_f32, )(pg, svundef_u32(), i64);
}

void test_f64(svbool_t pg, const float64_t *const_f64_ptr)
{
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_z'}}
  SVE_ACLE_FUNC(svlogb,_f64,_z,)(pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_m'}}
  SVE_ACLE_FUNC(svlogb,_f64,_m,)(svundef_s64(), pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svlogb_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svlogb_x'}}
  SVE_ACLE_FUNC(svlogb,_f64,_x,)(pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svminnmp_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminnmp_m'}}
  SVE_ACLE_FUNC(svminnmp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svminnmp_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminnmp_x'}}
  SVE_ACLE_FUNC(svminnmp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svtbl2_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbl2'}}
  SVE_ACLE_FUNC(svtbl2,_f64,,)(svundef2_f64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_m'}}
  SVE_ACLE_FUNC(svaddp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svaddp_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svaddp_x'}}
  SVE_ACLE_FUNC(svaddp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svtbx_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svtbx'}}
  SVE_ACLE_FUNC(svtbx,_f64,,)(svundef_f64(), svundef_f64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svminp_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_m'}}
  SVE_ACLE_FUNC(svminp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svminp_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svminp_x'}}
  SVE_ACLE_FUNC(svminp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, , _f64)(pg, svundef_u64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_s64offset_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, offset, _f64)(pg, f64_ptr, svundef_s64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64offset_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, offset, _f64)(pg, f64_ptr, svundef_u64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_offset_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_offset'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _offset, _f64)(pg, svundef_u64(), i64, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_s64index_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, s64, index, _f64)(pg, f64_ptr, svundef_s64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64index_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter_, u64, index, _f64)(pg, f64_ptr, svundef_u64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svstnt1_scatter_u64base_index_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svstnt1_scatter_index'}}
  SVE_ACLE_FUNC(svstnt1_scatter, _u64base, _index, _f64)(pg, svundef_u64(), i64, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_m'}}
  SVE_ACLE_FUNC(svmaxp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svmaxp_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxp_x'}}
  SVE_ACLE_FUNC(svmaxp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svmaxnmp_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxnmp_m'}}
  SVE_ACLE_FUNC(svmaxnmp,_f64,_m,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svmaxnmp_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svmaxnmp_x'}}
  SVE_ACLE_FUNC(svmaxnmp,_f64,_x,)(pg, svundef_f64(), svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svwhilerw_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilerw'}}
  SVE_ACLE_FUNC(svwhilerw,_f64,,)(const_f64_ptr, const_f64_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svcvtnt_f32_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtnt_f32_m'}}
  SVE_ACLE_FUNC(svcvtnt_f32,_f64,_m,)(svundef_f32(), pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svcvtnt_f32_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtnt_f32_x'}}
  SVE_ACLE_FUNC(svcvtnt_f32,_f64,_x,)(svundef_f32(), pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svwhilewr_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svwhilewr'}}
  SVE_ACLE_FUNC(svwhilewr,_f64,,)(const_f64_ptr, const_f64_ptr);
  // expected-error@+2 {{use of undeclared identifier 'svcvtx_f32_f64_z'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtx_f32_z'}}
  SVE_ACLE_FUNC(svcvtx_f32,_f64,_z,)(pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svcvtx_f32_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtx_f32_m'}}
  SVE_ACLE_FUNC(svcvtx_f32,_f64,_m,)(svundef_f32(), pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svcvtx_f32_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtx_f32_x'}}
  SVE_ACLE_FUNC(svcvtx_f32,_f64,_x,)(pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_f64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _f64, )(pg, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_s64offset_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _f64)(pg, const_f64_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64offset_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _f64)(pg, const_f64_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_offset_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_offset_f64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_f64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_s64index_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index'}}
  SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _f64)(pg, const_f64_ptr, svundef_s64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64index_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index'}}
  SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _f64)(pg, const_f64_ptr, svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svldnt1_gather_u64base_index_f64'}}
  // overload-error@+1 {{use of undeclared identifier 'svldnt1_gather_index_f64'}}
  SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_f64, )(pg, svundef_u64(), i64);
  // expected-error@+2 {{use of undeclared identifier 'svcvtxnt_f32_f64_m'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtxnt_f32_m'}}
  SVE_ACLE_FUNC(svcvtxnt_f32,_f64,_m,)(svundef_f32(), pg, svundef_f64());
  // expected-error@+2 {{use of undeclared identifier 'svcvtxnt_f32_f64_x'}}
  // overload-error@+1 {{use of undeclared identifier 'svcvtxnt_f32_x'}}
  SVE_ACLE_FUNC(svcvtxnt_f32,_f64,_x,)(svundef_f32(), pg, svundef_f64());
}
