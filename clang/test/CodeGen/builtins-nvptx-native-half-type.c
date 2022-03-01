// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu \
// RUN:   sm_80 -target-feature +ptx70 -fcuda-is-device -fnative-half-type -S \
// RUN:   -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX70_SM80 %s

// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown \
// RUN:   -target-cpu sm_80 -target-feature +ptx70 -fcuda-is-device \
// RUN:   -fnative-half-type -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX70_SM80 %s

// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu \
// RUN:   sm_86 -target-feature +ptx72 -fcuda-is-device -fnative-half-type -S \
// RUN:   -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX72_SM86 %s

// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown \
// RUN:   -target-cpu sm_86 -target-feature +ptx72 -fcuda-is-device \
// RUN:   -fnative-half-type -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX72_SM86 %s

// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu \
// RUN:   sm_53 -target-feature +ptx42 -fcuda-is-device -fnative-half-type -S \
// RUN:   -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX42_SM53 %s

// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown \
// RUN:   -target-cpu sm_53 -target-feature +ptx42 -fcuda-is-device \
// RUN:   -fnative-half-type -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX42_SM53 %s

#define __device__ __attribute__((device))

// CHECK-LABEL: nvvm_min_max_sm80
__device__ void nvvm_min_max_sm80() {
#if __CUDA_ARCH__ >= 800
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fmin.f16
  __nvvm_fmin_f16(0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fmin.ftz.f16
  __nvvm_fmin_ftz_f16(0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fmin.nan.f16
  __nvvm_fmin_nan_f16(0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fmin.ftz.nan.f16
  __nvvm_fmin_ftz_nan_f16(0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fmin.f16x2
  __nvvm_fmin_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fmin.ftz.f16x2
  __nvvm_fmin_ftz_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fmin.nan.f16x2
  __nvvm_fmin_nan_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fmin.ftz.nan.f16x2
  __nvvm_fmin_ftz_nan_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});

  // CHECK_PTX70_SM80: call half @llvm.nvvm.fmax.f16
  __nvvm_fmax_f16(0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fmax.ftz.f16
  __nvvm_fmax_ftz_f16(0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fmax.nan.f16
  __nvvm_fmax_nan_f16(0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fmax.ftz.nan.f16
  __nvvm_fmax_ftz_nan_f16(0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fmax.f16x2
  __nvvm_fmax_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fmax.ftz.f16x2
  __nvvm_fmax_ftz_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fmax.nan.f16x2
  __nvvm_fmax_nan_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fmax.ftz.nan.f16x2
  __nvvm_fmax_ftz_nan_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_fma_f16_f16x2_sm80
__device__ void nvvm_fma_f16_f16x2_sm80() {
#if __CUDA_ARCH__ >= 800
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fma.rn.relu.f16
  __nvvm_fma_rn_relu_f16(0.1f16, 0.1f16, 0.1f16);
  // CHECK_PTX70_SM80: call half @llvm.nvvm.fma.rn.ftz.relu.f16
  __nvvm_fma_rn_ftz_relu_f16(0.1f16, 0.1f16, 0.1f16);

  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fma.rn.relu.f16x2
  __nvvm_fma_rn_relu_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16},
                           {0.1f16, 0.7f16});
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.fma.rn.ftz.relu.f16x2
  __nvvm_fma_rn_ftz_relu_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16},
                               {0.1f16, 0.7f16});
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_fma_f16_f16x2_sm53
__device__ void nvvm_fma_f16_f16x2_sm53() {
#if __CUDA_ARCH__ >= 530
  // CHECK_PTX42_SM53: call half @llvm.nvvm.fma.rn.f16
  __nvvm_fma_rn_f16(0.1f16, 0.1f16, 0.1f16);
  // CHECK_PTX42_SM53: call half @llvm.nvvm.fma.rn.ftz.f16
  __nvvm_fma_rn_ftz_f16(0.1f16, 0.1f16, 0.1f16);
  // CHECK_PTX42_SM53: call half @llvm.nvvm.fma.rn.sat.f16
  __nvvm_fma_rn_sat_f16(0.1f16, 0.1f16, 0.1f16);
  // CHECK_PTX42_SM53: call half @llvm.nvvm.fma.rn.ftz.sat.f16
  __nvvm_fma_rn_ftz_sat_f16(0.1f16, 0.1f16, 0.1f16);

  // CHECK_PTX42_SM53: call <2 x half> @llvm.nvvm.fma.rn.f16x2
  __nvvm_fma_rn_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16},
                      {0.1f16, 0.7f16});
  // CHECK_PTX42_SM53: call <2 x half> @llvm.nvvm.fma.rn.ftz.f16x2
  __nvvm_fma_rn_ftz_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16},
                          {0.1f16, 0.7f16});
  // CHECK_PTX42_SM53: call <2 x half> @llvm.nvvm.fma.rn.sat.f16x2
  __nvvm_fma_rn_sat_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16},
                          {0.1f16, 0.7f16});
  // CHECK_PTX42_SM53: call <2 x half> @llvm.nvvm.fma.rn.ftz.sat.f16x2
  __nvvm_fma_rn_ftz_sat_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16},
                              {0.1f16, 0.7f16});
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_min_max_sm86
__device__ void nvvm_min_max_sm86() {
#if __CUDA_ARCH__ >= 860
  // CHECK_PTX72_SM86: call half @llvm.nvvm.fmin.xorsign.abs.f16
  __nvvm_fmin_xorsign_abs_f16(0.1f16, 0.1f16);
  // CHECK_PTX72_SM86: call half @llvm.nvvm.fmin.ftz.xorsign.abs.f16
  __nvvm_fmin_ftz_xorsign_abs_f16(0.1f16, 0.1f16);
  // CHECK_PTX72_SM86: call half @llvm.nvvm.fmin.nan.xorsign.abs.f16
  __nvvm_fmin_nan_xorsign_abs_f16(0.1f16, 0.1f16);
  // CHECK_PTX72_SM86: call half @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16
  __nvvm_fmin_ftz_nan_xorsign_abs_f16(0.1f16, 0.1f16);
  // CHECK_PTX72_SM86: call <2 x half> @llvm.nvvm.fmin.xorsign.abs.f16x2
  __nvvm_fmin_xorsign_abs_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX72_SM86: call <2 x half> @llvm.nvvm.fmin.ftz.xorsign.abs.f16x2
  __nvvm_fmin_ftz_xorsign_abs_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX72_SM86: call <2 x half> @llvm.nvvm.fmin.nan.xorsign.abs.f16x2
  __nvvm_fmin_nan_xorsign_abs_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX72_SM86: call <2 x half> @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16x2
  __nvvm_fmin_ftz_nan_xorsign_abs_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});

  // CHECK_PTX72_SM86: call half @llvm.nvvm.fmax.xorsign.abs.f16
  __nvvm_fmax_xorsign_abs_f16(0.1f16, 0.1f16);
  // CHECK_PTX72_SM86: call half @llvm.nvvm.fmax.ftz.xorsign.abs.f16
  __nvvm_fmax_ftz_xorsign_abs_f16(0.1f16, 0.1f16);
  // CHECK_PTX72_SM86: call half @llvm.nvvm.fmax.nan.xorsign.abs.f16
  __nvvm_fmax_nan_xorsign_abs_f16(0.1f16, 0.1f16);
  // CHECK_PTX72_SM86: call half @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16
  __nvvm_fmax_ftz_nan_xorsign_abs_f16(0.1f16, 0.1f16);
  // CHECK_PTX72_SM86: call <2 x half> @llvm.nvvm.fmax.xorsign.abs.f16x2
  __nvvm_fmax_xorsign_abs_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX72_SM86: call <2 x half> @llvm.nvvm.fmax.ftz.xorsign.abs.f16x2
  __nvvm_fmax_ftz_xorsign_abs_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX72_SM86: call <2 x half> @llvm.nvvm.fmax.nan.xorsign.abs.f16x2
  __nvvm_fmax_nan_xorsign_abs_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
  // CHECK_PTX72_SM86: call <2 x half> @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16x2
  __nvvm_fmax_ftz_nan_xorsign_abs_f16x2({0.1f16, 0.7f16}, {0.1f16, 0.7f16});
#endif
  // CHECK: ret void
}
