// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

void test_range_0_7(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqshlu,_n_s8,_x,)(pg, svundef_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqshlu,_n_s8,_z,)(pg, svundef_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqshlu,_n_s8,_m,)(pg, svundef_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svshllb,_n_s16,,)(svundef_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svshllb,_n_u16,,)(svundef_u8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svshllt,_n_s16,,)(svundef_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svshllt,_n_u16,,)(svundef_u8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svsli,_n_s8,,)(svundef_s8(), svundef_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svsli,_n_u8,,)(svundef_u8(), svundef_u8(), -1);
}

void test_range_1_8(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqrshrnb,_n_s16,,)(svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqrshrnb,_n_u16,,)(svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqrshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqrshrnt,_n_u16,,)(svundef_u8(), svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqrshrunb,_n_s16,,)(svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqrshrunt,_n_s16,,)(svundef_u8(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqshrnb,_n_s16,,)(svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqshrnb,_n_u16,,)(svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqshrnt,_n_u16,,)(svundef_u8(), svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqshrunb,_n_s16,,)(svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svqshrunt,_n_s16,,)(svundef_u8(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshr,_n_s8,_x,)(pg, svundef_s8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshr,_n_s8,_z,)(pg, svundef_s8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshr,_n_s8,_m,)(pg, svundef_s8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshr,_n_u8,_x,)(pg, svundef_u8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshr,_n_u8,_z,)(pg, svundef_u8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshr,_n_u8,_m,)(pg, svundef_u8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshrnb,_n_s16,,)(svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshrnb,_n_u16,,)(svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrshrnt,_n_u16,,)(svundef_u8(), svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrsra,_n_s8,,)(svundef_s8(), svundef_s8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svrsra,_n_u8,,)(svundef_u8(), svundef_u8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svshrnb,_n_s16,,)(svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svshrnb,_n_u16,,)(svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svshrnt,_n_s16,,)(svundef_s8(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svshrnt,_n_u16,,)(svundef_u8(), svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svsra,_n_s8,,)(svundef_s8(), svundef_s8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svsra,_n_u8,,)(svundef_u8(), svundef_u8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svsri,_n_s8,,)(svundef_s8(), svundef_s8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svsri,_n_u8,,)(svundef_u8(), svundef_u8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svxar,_n_s8,,)(svundef_s8(), svundef_s8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svxar,_n_u8,,)(svundef_u8(), svundef_u8(), 0);
}

void test_range_0_15(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svqshlu,_n_s16,_x,)(pg, svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svqshlu,_n_s16,_z,)(pg, svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svqshlu,_n_s16,_m,)(pg, svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svshllb,_n_s32,,)(svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svshllb,_n_u32,,)(svundef_u16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svshllt,_n_s32,,)(svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svshllt,_n_u32,,)(svundef_u16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svsli,_n_s16,,)(svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svsli,_n_u16,,)(svundef_u16(), svundef_u16(), -1);
}

void test_range_1_16(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqrshrnb,_n_s32,,)(svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqrshrnb,_n_u32,,)(svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqrshrnt,_n_s32,,)(svundef_s16(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqrshrnt,_n_u32,,)(svundef_u16(), svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqrshrunb,_n_s32,,)(svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqrshrunt,_n_s32,,)(svundef_u16(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqshrnb,_n_s32,,)(svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqshrnb,_n_u32,,)(svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqshrnt,_n_s32,,)(svundef_s16(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqshrnt,_n_u32,,)(svundef_u16(), svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqshrunb,_n_s32,,)(svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqshrunt,_n_s32,,)(svundef_u16(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshr,_n_s16,_x,)(pg, svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshr,_n_s16,_z,)(pg, svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshr,_n_s16,_m,)(pg, svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshr,_n_u16,_x,)(pg, svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshr,_n_u16,_z,)(pg, svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshr,_n_u16,_m,)(pg, svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshrnb,_n_s32,,)(svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshrnb,_n_u32,,)(svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshrnt,_n_s32,,)(svundef_s16(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrshrnt,_n_u32,,)(svundef_u16(), svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrsra,_n_s16,,)(svundef_s16(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svrsra,_n_u16,,)(svundef_u16(), svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svshrnb,_n_s32,,)(svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svshrnb,_n_u32,,)(svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svshrnt,_n_s32,,)(svundef_s16(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svshrnt,_n_u32,,)(svundef_u16(), svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svsra,_n_s16,,)(svundef_s16(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svsra,_n_u16,,)(svundef_u16(), svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svsri,_n_s16,,)(svundef_s16(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svsri,_n_u16,,)(svundef_u16(), svundef_u16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svxar,_n_s16,,)(svundef_s16(), svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svxar,_n_u16,,)(svundef_u16(), svundef_u16(), 0);
}

void test_range_0_31(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svqshlu,_n_s32,_x,)(pg, svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svqshlu,_n_s32,_z,)(pg, svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svqshlu,_n_s32,_m,)(pg, svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svshllb,_n_s64,,)(svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svshllb,_n_u64,,)(svundef_u32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svshllt,_n_s64,,)(svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svshllt,_n_u64,,)(svundef_u32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svsli,_n_s32,,)(svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svsli,_n_u32,,)(svundef_u32(), svundef_u32(), -1);
}

void test_range_1_32(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqrshrnb,_n_s64,,)(svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqrshrnb,_n_u64,,)(svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqrshrnt,_n_s64,,)(svundef_s32(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqrshrnt,_n_u64,,)(svundef_u32(), svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqrshrunb,_n_s64,,)(svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqrshrunt,_n_s64,,)(svundef_u32(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqshrnb,_n_s64,,)(svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqshrnb,_n_u64,,)(svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqshrnt,_n_s64,,)(svundef_s32(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqshrnt,_n_u64,,)(svundef_u32(), svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqshrunb,_n_s64,,)(svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svqshrunt,_n_s64,,)(svundef_u32(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshr,_n_s32,_x,)(pg, svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshr,_n_s32,_z,)(pg, svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshr,_n_s32,_m,)(pg, svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshr,_n_u32,_x,)(pg, svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshr,_n_u32,_z,)(pg, svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshr,_n_u32,_m,)(pg, svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshrnb,_n_s64,,)(svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshrnb,_n_u64,,)(svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshrnt,_n_s64,,)(svundef_s32(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrshrnt,_n_u64,,)(svundef_u32(), svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrsra,_n_s32,,)(svundef_s32(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svrsra,_n_u32,,)(svundef_u32(), svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svshrnb,_n_s64,,)(svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svshrnb,_n_u64,,)(svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svshrnt,_n_s64,,)(svundef_s32(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svshrnt,_n_u64,,)(svundef_u32(), svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svsra,_n_s32,,)(svundef_s32(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svsra,_n_u32,,)(svundef_u32(), svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svsri,_n_s32,,)(svundef_s32(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svsri,_n_u32,,)(svundef_u32(), svundef_u32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svxar,_n_s32,,)(svundef_s32(), svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svxar,_n_u32,,)(svundef_u32(), svundef_u32(), 0);
}

void test_range_0_63(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  SVE_ACLE_FUNC(svqshlu,_n_s64,_x,)(pg, svundef_s64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  SVE_ACLE_FUNC(svqshlu,_n_s64,_z,)(pg, svundef_s64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  SVE_ACLE_FUNC(svqshlu,_n_s64,_m,)(pg, svundef_s64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  SVE_ACLE_FUNC(svsli,_n_s64,,)(svundef_s64(), svundef_s64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  SVE_ACLE_FUNC(svsli,_n_u64,,)(svundef_u64(), svundef_u64(), -1);
}

void test_range_1_64(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svrshr,_n_s64,_x,)(pg, svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svrshr,_n_s64,_z,)(pg, svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svrshr,_n_s64,_m,)(pg, svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svrshr,_n_u64,_x,)(pg, svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svrshr,_n_u64,_z,)(pg, svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svrshr,_n_u64,_m,)(pg, svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svrsra,_n_s64,,)(svundef_s64(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svrsra,_n_u64,,)(svundef_u64(), svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svsra,_n_s64,,)(svundef_s64(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svsra,_n_u64,,)(svundef_u64(), svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svsri,_n_s64,,)(svundef_s64(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svsri,_n_u64,,)(svundef_u64(), svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svxar,_n_s64,,)(svundef_s64(), svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svxar,_n_u64,,)(svundef_u64(), svundef_u64(), 0);
}
