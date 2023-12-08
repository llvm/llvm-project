// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s

#include <arm_sve.h>

void test_bfloat(svbool_t pg, uint64_t u64, int64_t i64, const bfloat16_t *const_bf16_ptr, bfloat16_t *bf16_ptr, svbfloat16_t bf16, svbfloat16x2_t bf16x2, svbfloat16x3_t bf16x3, svbfloat16x4_t bf16x4)
{
  // expected-error@+1 {{'svcreate2_bf16' needs target feature sve,bf16}}
  svcreate2_bf16(bf16, bf16);
  // expected-error@+1 {{'svcreate3_bf16' needs target feature sve,bf16}}
  svcreate3_bf16(bf16, bf16, bf16);
  // expected-error@+1 {{'svcreate4_bf16' needs target feature sve,bf16}}
  svcreate4_bf16(bf16, bf16, bf16, bf16);
  // expected-error@+1 {{'svget2_bf16' needs target feature sve,bf16}}
  svget2_bf16(bf16x2, 1);
  // expected-error@+1 {{'svget3_bf16' needs target feature sve,bf16}}
  svget3_bf16(bf16x3, 1);
  // expected-error@+1 {{'svget4_bf16' needs target feature sve,bf16}}
  svget4_bf16(bf16x4, 1);
  // expected-error@+1 {{'svld1_bf16' needs target feature sve,bf16}}
  svld1_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{'svld1_vnum_bf16' needs target feature sve,bf16}}
  svld1_vnum_bf16(pg, const_bf16_ptr, i64);
  // expected-error@+1 {{'svld1rq_bf16' needs target feature sve,bf16}}
  svld1rq_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{'svldff1_bf16' needs target feature sve,bf16}}
  svldff1_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{'svldff1_vnum_bf16' needs target feature sve,bf16}}
  svldff1_vnum_bf16(pg, const_bf16_ptr, i64);
  // expected-error@+1 {{'svldnf1_bf16' needs target feature sve,bf16}}
  svldnf1_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{'svldnf1_vnum_bf16' needs target feature sve,bf16}}
  svldnf1_vnum_bf16(pg, const_bf16_ptr, i64);
  // expected-error@+1 {{'svldnt1_bf16' needs target feature sve,bf16}}
  svldnt1_bf16(pg, const_bf16_ptr);
  // expected-error@+1 {{'svldnt1_vnum_bf16' needs target feature sve,bf16}}
  svldnt1_vnum_bf16(pg, const_bf16_ptr, i64);
  // expected-error@+1 {{'svrev_bf16' needs target feature sve,bf16}}
  svrev_bf16(bf16);
  // expected-error@+1 {{'svset2_bf16' needs target feature sve,bf16}}
  svset2_bf16(bf16x2, 1, bf16);
  // expected-error@+1 {{'svset3_bf16' needs target feature sve,bf16}}
  svset3_bf16(bf16x3, 1, bf16);
  // expected-error@+1 {{'svset4_bf16' needs target feature sve,bf16}}
  svset4_bf16(bf16x4, 1, bf16);
  // expected-error@+1 {{'svst1_bf16' needs target feature sve,bf16}}
  svst1_bf16(pg, bf16_ptr, bf16);
  // expected-error@+1 {{'svst1_vnum_bf16' needs target feature sve,bf16}}
  svst1_vnum_bf16(pg, bf16_ptr, i64, bf16);
  // expected-error@+1 {{'svstnt1_bf16' needs target feature sve,bf16}}
  svstnt1_bf16(pg, bf16_ptr, bf16);
  // expected-error@+1 {{'svstnt1_vnum_bf16' needs target feature sve,bf16}}
  svstnt1_vnum_bf16(pg, bf16_ptr, i64, bf16);
  // expected-error@+1 {{'svtrn1_bf16' needs target feature sve,bf16}}
  svtrn1_bf16(bf16, bf16);
  // expected-error@+1 {{'svtrn1q_bf16' needs target feature sve,bf16}}
  svtrn1q_bf16(bf16, bf16);
  // expected-error@+1 {{'svtrn2_bf16' needs target feature sve,bf16}}
  svtrn2_bf16(bf16, bf16);
  // expected-error@+1 {{'svtrn2q_bf16' needs target feature sve,bf16}}
  svtrn2q_bf16(bf16, bf16);
  // expected-error@+1 {{'svundef_bf16' needs target feature sve,bf16}}
  svundef_bf16();
  // expected-error@+1 {{'svundef2_bf16' needs target feature sve,bf16}}
  svundef2_bf16();
  // expected-error@+1 {{'svundef3_bf16' needs target feature sve,bf16}}
  svundef3_bf16();
  // expected-error@+1 {{'svundef4_bf16' needs target feature sve,bf16}}
  svundef4_bf16();
  // expected-error@+1 {{'svuzp1_bf16' needs target feature sve,bf16}}
  svuzp1_bf16(bf16, bf16);
  // expected-error@+1 {{'svuzp1q_bf16' needs target feature sve,bf16}}
  svuzp1q_bf16(bf16, bf16);
  // expected-error@+1 {{'svuzp2_bf16' needs target feature sve,bf16}}
  svuzp2_bf16(bf16, bf16);
  // expected-error@+1 {{'svuzp2q_bf16' needs target feature sve,bf16}}
  svuzp2q_bf16(bf16, bf16);
  // expected-error@+1 {{'svzip1_bf16' needs target feature sve,bf16}}
  svzip1_bf16(bf16, bf16);
  // expected-error@+1 {{'svzip1q_bf16' needs target feature sve,bf16}}
  svzip1q_bf16(bf16, bf16);
  // expected-error@+1 {{'svzip2_bf16' needs target feature sve,bf16}}
  svzip2_bf16(bf16, bf16);
  // expected-error@+1 {{'svzip2q_bf16' needs target feature sve,bf16}}
  svzip2q_bf16(bf16, bf16);
}
