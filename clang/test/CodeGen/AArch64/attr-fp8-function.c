// Check memory attribute for FP8 function

// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -target-feature +fp8 -target-feature +sve -target-feature +sme -target-feature +sme2 -target-feature +sme-f8f16 -target-feature +sme-f8f32  -target-feature +ssve-fp8fma -disable-O0-optnone -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_neon.h>
#include <arm_sme.h>


// SIMD
mfloat8x16_t test_vcvtq_mf8_f16_fpm(float16x8_t vn, float16x8_t vm, fpm_t fpm) {
  return vcvtq_mf8_f16_fpm(vn, vm, fpm);
}

// SVE
svfloat16_t test_svcvtlt2_f16_mf8(svmfloat8_t zn, fpm_t fpm) __arm_streaming {
  return svcvtlt2_f16_mf8_fpm(zn, fpm);
}

// CHECK: declare void @llvm.aarch64.set.fpmr(i64) [[ATTR3:#.*]]
// CHECK: declare <vscale x 8 x half> @llvm.aarch64.sve.fp8.cvtlt2.nxv8f16(<vscale x 16 x i8>) [[ATTR4:#.*]]


// SME
svfloat32_t test_svmlalltt_lane_f32_mf8(svfloat32_t zda, svmfloat8_t zn, svmfloat8_t zm, fpm_t fpm) __arm_streaming {
  return svmlalltt_lane_f32_mf8_fpm(zda, zn, zm, 7, fpm);
}

// CHECK: declare <vscale x 4 x float> @llvm.aarch64.sve.fp8.fmlalltt.lane.nxv4f32(<vscale x 4 x float>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32 immarg) [[ATTR4]]

// CHECK: declare <16 x i8> @llvm.aarch64.neon.fp8.fcvtn.v16i8.v8f16(<8 x half>, <8 x half>) [[ATTR4]]

// CHECK: attributes [[ATTR1:#.*]] = {{{.*}}} 
// CHECK: attributes [[ATTR2:#.*]] = {{{.*}}}
// CHECK: attributes [[ATTR3]] = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
// CHECK: attributes [[ATTR4]] = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
