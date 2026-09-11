// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90a \
// RUN:            -target-feature +ptx87 -x cuda -fcuda-is-device -fclangir -emit-cir \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90a \
// RUN:            -target-feature +ptx87 -x cuda -fcuda-is-device -fclangir -emit-llvm \
// RUN:            %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90a \
// RUN:            -target-feature +ptx87 -x cuda -fcuda-is-device -emit-llvm \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR
// calls yet, so LLVM check lines use {{.*}} to tolerate the difference
// between CIR (no flags) and classic codegen ('contract').

#define __device__ __attribute__((device))

typedef __fp16 f16x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

// CIR-LABEL: @test_ex2_approx_f16
// CIR: cir.call_llvm_intrinsic "nvvm.ex2.approx" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM-LABEL: @test_ex2_approx_f16(
// LLVM: call{{.*}} half @llvm.nvvm.ex2.approx.f16(half %{{.*}})
extern "C" __device__ _Float16 test_ex2_approx_f16(_Float16 a0) {
  return __nvvm_ex2_approx_f16(a0);
}

// CIR-LABEL: @test_ex2_approx_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.ex2.approx" {{.*}} : (!cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_ex2_approx_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.ex2.approx.v2f16(<2 x half> %{{.*}})
extern "C" __device__ f16x2 test_ex2_approx_f16x2(f16x2 a0) {
  return __nvvm_ex2_approx_f16x2(a0);
}

// CIR-LABEL: @test_fma_rn_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.f16" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fma_rn_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fma.rn.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fma_rn_f16(_Float16 a0, _Float16 a1, _Float16 a2) {
  return __nvvm_fma_rn_f16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fma_rn_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fma.rn.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fma_rn_f16x2(f16x2 a0, f16x2 a1, f16x2 a2) {
  return __nvvm_fma_rn_f16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_ftz_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.ftz.f16" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fma_rn_ftz_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fma.rn.ftz.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fma_rn_ftz_f16(_Float16 a0, _Float16 a1, _Float16 a2) {
  return __nvvm_fma_rn_ftz_f16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_ftz_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.ftz.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fma_rn_ftz_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fma.rn.ftz.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fma_rn_ftz_f16x2(f16x2 a0, f16x2 a1, f16x2 a2) {
  return __nvvm_fma_rn_ftz_f16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_ftz_relu_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.ftz.relu.f16" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fma_rn_ftz_relu_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fma.rn.ftz.relu.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fma_rn_ftz_relu_f16(_Float16 a0, _Float16 a1, _Float16 a2) {
  return __nvvm_fma_rn_ftz_relu_f16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_ftz_relu_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.ftz.relu.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fma_rn_ftz_relu_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fma.rn.ftz.relu.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fma_rn_ftz_relu_f16x2(f16x2 a0, f16x2 a1, f16x2 a2) {
  return __nvvm_fma_rn_ftz_relu_f16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_ftz_sat_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.ftz.sat.f16" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fma_rn_ftz_sat_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fma.rn.ftz.sat.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fma_rn_ftz_sat_f16(_Float16 a0, _Float16 a1, _Float16 a2) {
  return __nvvm_fma_rn_ftz_sat_f16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_ftz_sat_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.ftz.sat.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fma_rn_ftz_sat_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fma.rn.ftz.sat.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fma_rn_ftz_sat_f16x2(f16x2 a0, f16x2 a1, f16x2 a2) {
  return __nvvm_fma_rn_ftz_sat_f16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_relu_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.relu.f16" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fma_rn_relu_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fma.rn.relu.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fma_rn_relu_f16(_Float16 a0, _Float16 a1, _Float16 a2) {
  return __nvvm_fma_rn_relu_f16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_relu_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.relu.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fma_rn_relu_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fma.rn.relu.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fma_rn_relu_f16x2(f16x2 a0, f16x2 a1, f16x2 a2) {
  return __nvvm_fma_rn_relu_f16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_sat_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.sat.f16" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fma_rn_sat_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fma.rn.sat.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fma_rn_sat_f16(_Float16 a0, _Float16 a1, _Float16 a2) {
  return __nvvm_fma_rn_sat_f16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_sat_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.sat.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fma_rn_sat_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fma.rn.sat.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fma_rn_sat_f16x2(f16x2 a0, f16x2 a1, f16x2 a2) {
  return __nvvm_fma_rn_sat_f16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fmax_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmax_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmax.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmax_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmax_f16(a0, a1);
}

// CIR-LABEL: @test_fmax_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmax_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmax.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmax_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmax_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmax_ftz_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.ftz.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmax_ftz_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmax.ftz.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmax_ftz_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmax_ftz_f16(a0, a1);
}

// CIR-LABEL: @test_fmax_ftz_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.ftz.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmax_ftz_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmax.ftz.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmax_ftz_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmax_ftz_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmax_ftz_nan_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.ftz.nan.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmax_ftz_nan_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmax.ftz.nan.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmax_ftz_nan_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmax_ftz_nan_f16(a0, a1);
}

// CIR-LABEL: @test_fmax_ftz_nan_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.ftz.nan.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmax_ftz_nan_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmax.ftz.nan.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmax_ftz_nan_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmax_ftz_nan_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmax_ftz_nan_xorsign_abs_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.ftz.nan.xorsign.abs.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmax_ftz_nan_xorsign_abs_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmax_ftz_nan_xorsign_abs_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmax_ftz_nan_xorsign_abs_f16(a0, a1);
}

// CIR-LABEL: @test_fmax_ftz_nan_xorsign_abs_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.ftz.nan.xorsign.abs.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmax_ftz_nan_xorsign_abs_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmax_ftz_nan_xorsign_abs_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmax_ftz_nan_xorsign_abs_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmax_ftz_xorsign_abs_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.ftz.xorsign.abs.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmax_ftz_xorsign_abs_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmax.ftz.xorsign.abs.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmax_ftz_xorsign_abs_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmax_ftz_xorsign_abs_f16(a0, a1);
}

// CIR-LABEL: @test_fmax_ftz_xorsign_abs_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.ftz.xorsign.abs.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmax_ftz_xorsign_abs_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmax.ftz.xorsign.abs.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmax_ftz_xorsign_abs_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmax_ftz_xorsign_abs_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmax_nan_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.nan.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmax_nan_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmax.nan.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmax_nan_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmax_nan_f16(a0, a1);
}

// CIR-LABEL: @test_fmax_nan_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.nan.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmax_nan_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmax.nan.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmax_nan_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmax_nan_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmax_nan_xorsign_abs_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.nan.xorsign.abs.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmax_nan_xorsign_abs_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmax.nan.xorsign.abs.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmax_nan_xorsign_abs_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmax_nan_xorsign_abs_f16(a0, a1);
}

// CIR-LABEL: @test_fmax_nan_xorsign_abs_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.nan.xorsign.abs.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmax_nan_xorsign_abs_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmax.nan.xorsign.abs.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmax_nan_xorsign_abs_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmax_nan_xorsign_abs_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmax_xorsign_abs_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.xorsign.abs.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmax_xorsign_abs_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmax.xorsign.abs.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmax_xorsign_abs_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmax_xorsign_abs_f16(a0, a1);
}

// CIR-LABEL: @test_fmax_xorsign_abs_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmax.xorsign.abs.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmax_xorsign_abs_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmax.xorsign.abs.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmax_xorsign_abs_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmax_xorsign_abs_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmin_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmin_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmin.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmin_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmin_f16(a0, a1);
}

// CIR-LABEL: @test_fmin_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmin_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmin.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmin_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmin_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmin_ftz_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.ftz.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmin_ftz_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmin.ftz.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmin_ftz_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmin_ftz_f16(a0, a1);
}

// CIR-LABEL: @test_fmin_ftz_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.ftz.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmin_ftz_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmin.ftz.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmin_ftz_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmin_ftz_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmin_ftz_nan_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.ftz.nan.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmin_ftz_nan_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmin.ftz.nan.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmin_ftz_nan_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmin_ftz_nan_f16(a0, a1);
}

// CIR-LABEL: @test_fmin_ftz_nan_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.ftz.nan.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmin_ftz_nan_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmin.ftz.nan.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmin_ftz_nan_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmin_ftz_nan_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmin_ftz_nan_xorsign_abs_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.ftz.nan.xorsign.abs.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmin_ftz_nan_xorsign_abs_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmin_ftz_nan_xorsign_abs_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmin_ftz_nan_xorsign_abs_f16(a0, a1);
}

// CIR-LABEL: @test_fmin_ftz_nan_xorsign_abs_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.ftz.nan.xorsign.abs.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmin_ftz_nan_xorsign_abs_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmin_ftz_nan_xorsign_abs_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmin_ftz_nan_xorsign_abs_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmin_ftz_xorsign_abs_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.ftz.xorsign.abs.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmin_ftz_xorsign_abs_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmin.ftz.xorsign.abs.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmin_ftz_xorsign_abs_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmin_ftz_xorsign_abs_f16(a0, a1);
}

// CIR-LABEL: @test_fmin_ftz_xorsign_abs_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.ftz.xorsign.abs.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmin_ftz_xorsign_abs_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmin.ftz.xorsign.abs.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmin_ftz_xorsign_abs_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmin_ftz_xorsign_abs_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmin_nan_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.nan.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmin_nan_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmin.nan.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmin_nan_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmin_nan_f16(a0, a1);
}

// CIR-LABEL: @test_fmin_nan_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.nan.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmin_nan_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmin.nan.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmin_nan_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmin_nan_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmin_nan_xorsign_abs_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.nan.xorsign.abs.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmin_nan_xorsign_abs_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmin.nan.xorsign.abs.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmin_nan_xorsign_abs_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmin_nan_xorsign_abs_f16(a0, a1);
}

// CIR-LABEL: @test_fmin_nan_xorsign_abs_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.nan.xorsign.abs.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmin_nan_xorsign_abs_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmin.nan.xorsign.abs.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmin_nan_xorsign_abs_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmin_nan_xorsign_abs_f16x2(a0, a1);
}

// CIR-LABEL: @test_fmin_xorsign_abs_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.xorsign.abs.f16" {{.*}} : (!cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fmin_xorsign_abs_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fmin.xorsign.abs.f16(half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fmin_xorsign_abs_f16(_Float16 a0, _Float16 a1) {
  return __nvvm_fmin_xorsign_abs_f16(a0, a1);
}

// CIR-LABEL: @test_fmin_xorsign_abs_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fmin.xorsign.abs.f16x2" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fmin_xorsign_abs_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fmin.xorsign.abs.f16x2(<2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fmin_xorsign_abs_f16x2(f16x2 a0, f16x2 a1) {
  return __nvvm_fmin_xorsign_abs_f16x2(a0, a1);
}
