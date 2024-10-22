// REQUIRES: nvptx-registered-target
//
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu \
// RUN:   sm_86 -target-feature +ptx72 -fcuda-is-device -x cuda -emit-llvm -o - %s \
// RUN:   | FileCheck %s

#define __device__ __attribute__((device))
typedef __fp16 __fp16v2 __attribute__((ext_vector_type(2)));

// CHECK: call half @llvm.nvvm.ex2.approx.f16(half {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.ex2.approx.f16x2(<2 x half> {{.*}})
// CHECK: call half @llvm.nvvm.fma.rn.relu.f16(half {{.*}}, half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fma.rn.ftz.relu.f16(half {{.*}}, half {{.*}}, half {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fma.rn.relu.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fma.rn.ftz.relu.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call half @llvm.nvvm.fma.rn.ftz.f16(half {{.*}}, half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fma.rn.sat.f16(half {{.*}}, half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fma.rn.ftz.sat.f16(half {{.*}}, half {{.*}}, half {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fma.rn.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fma.rn.ftz.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fma.rn.sat.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fma.rn.ftz.sat.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call half @llvm.nvvm.fmin.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmin.ftz.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmin.nan.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmin.ftz.nan.f16(half {{.*}}, half {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmin.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmin.ftz.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmin.nan.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmin.ftz.nan.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call half @llvm.nvvm.fmin.xorsign.abs.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmin.ftz.xorsign.abs.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmin.nan.xorsign.abs.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16(half {{.*}}, half {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmin.xorsign.abs.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmin.ftz.xorsign.abs.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmin.nan.xorsign.abs.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call half @llvm.nvvm.fmax.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmax.ftz.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmax.nan.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmax.ftz.nan.f16(half {{.*}}, half {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmax.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmax.ftz.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmax.nan.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmax.ftz.nan.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call half @llvm.nvvm.fmax.xorsign.abs.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmax.ftz.xorsign.abs.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmax.nan.xorsign.abs.f16(half {{.*}}, half {{.*}})
// CHECK: call half @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16(half {{.*}}, half {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmax.xorsign.abs.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmax.ftz.xorsign.abs.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmax.nan.xorsign.abs.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call <2 x half> @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16x2(<2 x half> {{.*}}, <2 x half> {{.*}})
// CHECK: call half @llvm.nvvm.ldg.global.f.f16.p0(ptr {{.*}}, i32 2)
// CHECK: call <2 x half> @llvm.nvvm.ldg.global.f.v2f16.p0(ptr {{.*}}, i32 4)
// CHECK: call half @llvm.nvvm.ldu.global.f.f16.p0(ptr {{.*}}, i32 2)
// CHECK: call <2 x half> @llvm.nvvm.ldu.global.f.v2f16.p0(ptr {{.*}}, i32 4)
__device__ void nvvm_native_half_types(void *a, void*b, void*c, __fp16* out) {
  __fp16v2 resv2 = {0, 0};
  *out += __nvvm_ex2_approx_f16(*(__fp16 *)a);
  resv2 = __nvvm_ex2_approx_f16x2(*(__fp16v2*)a);

  *out += __nvvm_fma_rn_relu_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16*)c);
  *out += __nvvm_fma_rn_ftz_relu_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16 *)c);
  resv2 += __nvvm_fma_rn_relu_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  resv2 += __nvvm_fma_rn_ftz_relu_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  *out += __nvvm_fma_rn_ftz_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16*)c);
  *out += __nvvm_fma_rn_sat_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16*)c);
  *out += __nvvm_fma_rn_ftz_sat_f16(*(__fp16*)a, *(__fp16*)b, *(__fp16*)c);
  resv2 += __nvvm_fma_rn_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  resv2 += __nvvm_fma_rn_ftz_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  resv2 += __nvvm_fma_rn_sat_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);
  resv2 += __nvvm_fma_rn_ftz_sat_f16x2(*(__fp16v2*)a, *(__fp16v2*)b, *(__fp16v2*)c);

  *out += __nvvm_fmin_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_ftz_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_nan_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_ftz_nan_f16(*(__fp16*)a, *(__fp16*)b);
  resv2 += __nvvm_fmin_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmin_ftz_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmin_nan_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmin_ftz_nan_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  *out += __nvvm_fmin_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_ftz_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_nan_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmin_ftz_nan_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  resv2 += __nvvm_fmin_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmin_ftz_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmin_nan_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmin_ftz_nan_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);

  *out += __nvvm_fmax_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_ftz_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_nan_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_ftz_nan_f16(*(__fp16*)a, *(__fp16*)b);
  resv2 += __nvvm_fmax_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmax_ftz_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmax_nan_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  resv2 += __nvvm_fmax_ftz_nan_f16x2(*(__fp16v2*)a , *(__fp16v2*)b);
  *out += __nvvm_fmax_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_ftz_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_nan_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  *out += __nvvm_fmax_ftz_nan_xorsign_abs_f16(*(__fp16*)a, *(__fp16*)b);
  resv2 += __nvvm_fmax_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmax_ftz_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmax_nan_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);
  resv2 += __nvvm_fmax_ftz_nan_xorsign_abs_f16x2(*(__fp16v2*)a, *(__fp16v2*)b);

  *out += __nvvm_ldg_h((__fp16 *)a);
  resv2 += __nvvm_ldg_h2((__fp16v2 *)a);

  *out += __nvvm_ldu_h((__fp16 *)a);
  resv2 += __nvvm_ldu_h2((__fp16v2 *)a);

  *out += resv2[0] + resv2[1];
}
