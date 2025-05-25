// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -verify -emit-llvm -o - -ferror-limit 100 %s
// REQUIRES: aarch64-registered-target

// Test that functions with the correct target attributes can use the correct SVE intrinsics.
// expected-no-diagnostics

#include <arm_sve.h>

void __attribute__((target("sve"))) test_sve(svint64_t x, svint64_t y)
{
  svzip2(x, y);
}

void __attribute__((target("sve,bf16"))) test_bfloat(svfloat32_t x, svbfloat16_t y, bfloat16_t z)
{
  svbfdot_n_f32(x, y, z);
}

void __attribute__((target("sve2"))) test_sve2(svbool_t pg)
{
  svlogb_f16_z(pg, svundef_f16());
}

void __attribute__((target("sve2-sha3"))) test_sve2_sha3()
{
  svrax1_s64(svundef_s64(), svundef_s64());
}

void __attribute__((target("sve2"))) test_f16(svbool_t pg)
{
  svlogb_f16_z(pg, svundef_f16());
}

