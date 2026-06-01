// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-unknown-linux-gnu -emit-llvm %s \
// RUN:            -Werror \
// RUN:            -target-feature +msa -target-feature +fp64 \
// RUN:            -mfloat-abi hard -o - | FileCheck %s

#include <msa.h>

void test(void) {
  const v16i8 v16i8_a = (v16i8) {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  v16i8 v16i8_r;
  const v8i16 v8i16_a = (v8i16) {0, 1, 2, 3, 4, 5, 6, 7};
  v8i16 v8i16_r;
  const v4i32 v4i32_a = (v4i32) {0, 1, 2, 3};
  v4i32 v4i32_r;
  const v2i64 v2i64_a = (v2i64) {0, 1};
  v2i64 v2i64_r;

  v16i8_r = __msa_ld_b(&v16i8_a, 1); // CHECK: call <16 x i8>  @llvm.mips.ld.b(
  v8i16_r = __msa_ld_h(&v8i16_a, 2); // CHECK: call <8  x i16> @llvm.mips.ld.h(
  v4i32_r = __msa_ld_w(&v4i32_a, 4); // CHECK: call <4  x i32> @llvm.mips.ld.w(
  v2i64_r = __msa_ld_d(&v2i64_a, 8); // CHECK: call <2  x i64> @llvm.mips.ld.d(

}
