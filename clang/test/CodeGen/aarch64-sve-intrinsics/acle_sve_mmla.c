// RUN: %clang_cc1 -target-feature +i8mm -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -target-feature +i8mm -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -target-feature +i8mm -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -target-feature +i8mm -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svint32_t test_svmmla_s32(svint32_t x, svint8_t y, svint8_t z) {
  // CHECK-LABEL: test_svmmla_s32
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.smmla.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %z)
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svmmla,_s32,,)(x, y, z);
}

svuint32_t test_svmmla_u32(svuint32_t x, svuint8_t y, svuint8_t z) {
  // CHECK-LABEL: test_svmmla_u32
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ummla.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %z)
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svmmla,_u32,,)(x, y, z);
}

svint32_t test_svusmmla_s32(svint32_t x, svuint8_t y, svint8_t z) {
  // CHECK-LABEL: test_svusmmla_s32
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usmmla.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %z)
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svusmmla,_s32,,)(x, y, z);
}
