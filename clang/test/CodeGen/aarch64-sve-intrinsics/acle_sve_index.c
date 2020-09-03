// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
#include <arm_sve.h>

svint8_t test_svindex_s8(int8_t base, int8_t step)
{
  // CHECK-LABEL: test_svindex_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.index.nxv16i8(i8 %base, i8 %step)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return svindex_s8(base, step);
}

svint16_t test_svindex_s16(int16_t base, int16_t step)
{
  // CHECK-LABEL: test_svindex_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.index.nxv8i16(i16 %base, i16 %step)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return svindex_s16(base, step);
}

svint32_t test_svindex_s32(int32_t base, int32_t step)
{
  // CHECK-LABEL: test_svindex_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.index.nxv4i32(i32 %base, i32 %step)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return svindex_s32(base, step);
}

svint64_t test_svindex_s64(int64_t base, int64_t step)
{
  // CHECK-LABEL: test_svindex_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.index.nxv2i64(i64 %base, i64 %step)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return svindex_s64(base, step);
}

svuint8_t test_svindex_u8(uint8_t base, uint8_t step)
{
  // CHECK-LABEL: test_svindex_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.index.nxv16i8(i8 %base, i8 %step)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return svindex_u8(base, step);
}

svuint16_t test_svindex_u16(uint16_t base, uint16_t step)
{
  // CHECK-LABEL: test_svindex_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.index.nxv8i16(i16 %base, i16 %step)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return svindex_u16(base, step);
}

svuint32_t test_svindex_u32(uint32_t base, uint32_t step)
{
  // CHECK-LABEL: test_svindex_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.index.nxv4i32(i32 %base, i32 %step)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return svindex_u32(base, step);
}

svuint64_t test_svindex_u64(uint64_t base, uint64_t step)
{
  // CHECK-LABEL: test_svindex_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.index.nxv2i64(i64 %base, i64 %step)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return svindex_u64(base, step);
}
