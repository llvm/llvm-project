#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR calls
// yet, so LLVM check lines use {{.*}} to tolerate the difference between
// CIR (no flags) and classic codegen ('contract').

// CIR-LABEL: @_Z11test_fabs_ff
// CIR: cir.call_llvm_intrinsic "nvvm.fabs" {{.*}} : (!cir.float) -> !cir.float
// LLVM-LABEL: @_Z11test_fabs_ff
// LLVM: call {{.*}}float @llvm.nvvm.fabs.f32(float
__device__ float test_fabs_f(float x) {
  return __nvvm_fabs_f(x);
}

// CIR-LABEL: @_Z15test_fabs_ftz_ff
// CIR: cir.call_llvm_intrinsic "nvvm.fabs.ftz" {{.*}} : (!cir.float) -> !cir.float
// LLVM-LABEL: @_Z15test_fabs_ftz_ff
// LLVM: call {{.*}}float @llvm.nvvm.fabs.ftz.f32(float
__device__ float test_fabs_ftz_f(float x) {
  return __nvvm_fabs_ftz_f(x);
}

// CIR-LABEL: @_Z11test_fabs_dd
// CIR: cir.call_llvm_intrinsic "fabs" {{.*}} : (!cir.double) -> !cir.double
// LLVM-LABEL: @_Z11test_fabs_dd
// LLVM: call {{.*}}double @llvm.fabs.f64(double
__device__ double test_fabs_d(double x) {
  return __nvvm_fabs_d(x);
}

// CIR-LABEL: @_Z17test_ex2_approx_ff
// CIR: cir.call_llvm_intrinsic "nvvm.ex2.approx" {{.*}} : (!cir.float) -> !cir.float
// LLVM-LABEL: @_Z17test_ex2_approx_ff
// LLVM: call {{.*}}float @llvm.nvvm.ex2.approx.f32(float
__device__ float test_ex2_approx_f(float x) {
  return __nvvm_ex2_approx_f(x);
}

// CIR-LABEL: @_Z17test_ex2_approx_dd
// CIR: cir.call_llvm_intrinsic "nvvm.ex2.approx" {{.*}} : (!cir.double) -> !cir.double
// LLVM-LABEL: @_Z17test_ex2_approx_dd
// LLVM: call {{.*}}double @llvm.nvvm.ex2.approx.f64(double
__device__ double test_ex2_approx_d(double x) {
  return __nvvm_ex2_approx_d(x);
}

// CIR-LABEL: @_Z21test_ex2_approx_ftz_ff
// CIR: cir.call_llvm_intrinsic "nvvm.ex2.approx.ftz" {{.*}} : (!cir.float) -> !cir.float
// LLVM-LABEL: @_Z21test_ex2_approx_ftz_ff
// LLVM: call {{.*}}float @llvm.nvvm.ex2.approx.ftz.f32(float
__device__ float test_ex2_approx_ftz_f(float x) {
  return __nvvm_ex2_approx_ftz_f(x);
}

// CIR-LABEL: @_Z13test_add_rn_fff
// CIR: cir.call_llvm_intrinsic "nvvm.fadd" {{.*}} : (!cir.float, !cir.float, !s32i) -> !cir.float
// LLVM-LABEL: @_Z13test_add_rn_fff
// LLVM: call {{.*}}float @llvm.nvvm.fadd.f32(float {{.*}}, float {{.*}}, /* rnd=rn */ i32 1)
__device__ float test_add_rn_f(float x, float y) {
  return __nvvm_add_rn_f(x, y);
}

// CIR-LABEL: @_Z13test_add_rz_ddd
// CIR: cir.call_llvm_intrinsic "nvvm.fadd" {{.*}} : (!cir.double, !cir.double, !s32i) -> !cir.double
// LLVM-LABEL: @_Z13test_add_rz_ddd
// LLVM: call {{.*}}double @llvm.nvvm.fadd.f64(double {{.*}}, double {{.*}}, /* rnd=rz */ i32 0)
__device__ double test_add_rz_d(double x, double y) {
  return __nvvm_add_rz_d(x, y);
}

// CIR-LABEL: @_Z21test_add_rm_ftz_sat_fff
// CIR: cir.call_llvm_intrinsic "nvvm.fadd.ftz.sat" {{.*}} : (!cir.float, !cir.float, !s32i) -> !cir.float
// LLVM-LABEL: @_Z21test_add_rm_ftz_sat_fff
// LLVM: call {{.*}}float @llvm.nvvm.fadd.ftz.sat.f32(float {{.*}}, float {{.*}}, /* rnd=rm */ i32 3)
__device__ float test_add_rm_ftz_sat_f(float x, float y) {
  return __nvvm_add_rm_ftz_sat_f(x, y);
}
