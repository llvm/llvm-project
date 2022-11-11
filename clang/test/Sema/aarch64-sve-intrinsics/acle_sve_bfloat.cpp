// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify -verify-ignore-unexpected=error,note %s

#include <arm_sve.h>

void test_bfloat(svbool_t pg, uint64_t u64, int64_t i64, const bfloat16_t *const_bf16_ptr, svbfloat16_t bf16, svbfloat16x2_t bf16x2, svbfloat16x3_t bf16x3, svbfloat16x4_t bf16x4)
{
  // expected-error@+1 {{use of undeclared identifier 'svcreate2_bf16'}}
  svcreate2_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svcreate3_bf16'}}
  svcreate3_bf16(bf16, bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svcreate4_bf16'}}
  svcreate4_bf16(bf16, bf16, bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svget2_bf16'}}
  svget2_bf16(bf16x2, u64);
  // expected-error@+1 {{use of undeclared identifier 'svget3_bf16'}}
  svget3_bf16(bf16x3, u64);
  // expected-error@+1 {{use of undeclared identifier 'svget4_bf16'}}
  svget4_bf16(bf16x4, u64);
  // expected-error@+1 {{use of undeclared identifier 'svld1_bf16'}}
  svld1_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{use of undeclared identifier 'svld1_vnum_bf16'}}
  svld1_vnum_bf16(pg, const_bf16_ptr, i64);
  // expected-error@+1 {{use of undeclared identifier 'svld1rq_bf16'}}
  svld1rq_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{use of undeclared identifier 'svldff1_bf16'}}
  svldff1_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{use of undeclared identifier 'svldff1_vnum_bf16'}}
  svldff1_vnum_bf16(pg, const_bf16_ptr, i64);
  // expected-error@+1 {{use of undeclared identifier 'svldnf1_bf16'}}
  svldnf1_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{use of undeclared identifier 'svldnf1_vnum_bf16'}}
  svldnf1_vnum_bf16(pg, const_bf16_ptr, i64);
  // expected-error@+1 {{use of undeclared identifier 'svldnt1_bf16'}}
  svldnt1_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{use of undeclared identifier 'svldnt1_vnum_bf16'}}
  svldnt1_vnum_bf16(pg, const_bf16_ptr, i64);
  // expected-error@+1 {{use of undeclared identifier 'svrev_bf16'}}
  svrev_bf16(bf16);
  // expected-error@+1 {{use of undeclared identifier 'svset2_bf16'}}
  svset2_bf16(bf16x2, u64, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svset3_bf16'}}
  svset3_bf16(bf16x3, u64, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svset4_bf16'}}
  svset4_bf16(bf16x4, u64, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svst1_bf16'}}
  svst1_bf16(pg, const_bf16_ptr, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svst1_vnum_bf16'}}
  svst1_vnum_bf16(pg, const_bf16_ptr, i64, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svstnt1_bf16'}}
  svstnt1_bf16(pg, const_bf16_ptr, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svstnt1_vnum_bf16'}}
  svstnt1_vnum_bf16(pg, const_bf16_ptr, i64, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svtrn1_bf16'}}
  svtrn1_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svtrn1q_bf16'}}
  svtrn1q_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svtrn2_bf16'}}
  svtrn2_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svtrn2q_bf16'}}
  svtrn2q_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svundef_bf16'}}
  svundef_bf16();
  // expected-error@+1 {{use of undeclared identifier 'svundef2_bf16'}}
  svundef2_bf16();
  // expected-error@+1 {{use of undeclared identifier 'svundef3_bf16'}}
  svundef3_bf16();
  // expected-error@+1 {{use of undeclared identifier 'svundef4_bf16'}}
  svundef4_bf16();
  // expected-error@+1 {{use of undeclared identifier 'svuzp1_bf16'}}
  svuzp1_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svuzp1q_bf16'}}
  svuzp1q_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svuzp2_bf16'}}
  svuzp2_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svuzp2q_bf16'}}
  svuzp2q_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svzip1_bf16'}}
  svzip1_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svzip1q_bf16'}}
  svzip1q_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svzip2_bf16'}}
  svzip2_bf16(bf16, bf16);
  // expected-error@+1 {{use of undeclared identifier 'svzip2q_bf16'}}
  svzip2q_bf16(bf16, bf16);
}
