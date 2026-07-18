// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#define __device__ __attribute__((device))

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z18test_div_scale_f64PdPidd
// CIR: cir.call_llvm_intrinsic "amdgcn.div.scale" {{.*}} : (!cir.double, !cir.double, !cir.bool)
// LLVM: define{{.*}} void @_Z18test_div_scale_f64PdPidd
// LLVM: call { double, i1 } @llvm.amdgcn.div.scale.f64(double %{{.+}}, double %{{.+}}, i1 true)
__device__ void test_div_scale_f64(double* out, int* flagout, double a, double b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scale(a, b, true, &flag);
  *flagout = flag;
}

// CIR-LABEL: @_Z18test_div_scale_f32PfPbff
// CIR: cir.call_llvm_intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @_Z18test_div_scale_f32PfPbff
// LLVM: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
__device__ void test_div_scale_f32(float* out, bool* flagout, float a, float b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scalef(a, b, true, &flag);
  *flagout = flag;
}

// CIR-LABEL: @_Z27test_div_scale_f32_with_ptrPfPiPbff
// CIR: cir.call_llvm_intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @_Z27test_div_scale_f32_with_ptrPfPiPbff
// LLVM: {{.*}}call{{.*}} { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
__device__ void test_div_scale_f32_with_ptr(float* out, int* flagout, bool* flag, float a, float b)
{
  *out = __builtin_amdgcn_div_scalef(a, b, true, flag);
}

// CIR-LABEL: @_Z17test_div_fmas_f32Pdfffi
// CIR: cir.call_llvm_intrinsic "amdgcn.div.fmas" {{.*}} : (!cir.float, !cir.float, !cir.float, !cir.bool) -> !cir.float
// LLVM: define{{.*}} void @_Z17test_div_fmas_f32Pdfffi
// LLVM: call{{.*}} float @llvm.amdgcn.div.fmas.f32(float %{{.+}}, float %{{.+}}, float %{{.+}}, i1 %{{.*}})
__device__ void test_div_fmas_f32(double* out, float a, float b, float c, int d) {
  *out = __builtin_amdgcn_div_fmasf(a, b, c, d);
}

// CIR-LABEL: @_Z17test_div_fmas_f64Pddddi
// CIR: cir.call_llvm_intrinsic "amdgcn.div.fmas" {{.*}} : (!cir.double, !cir.double, !cir.double, !cir.bool) -> !cir.double
// LLVM: define{{.*}} void @_Z17test_div_fmas_f64Pddddi
// LLVM: call{{.*}} double @llvm.amdgcn.div.fmas.f64(double %{{.+}}, double %{{.+}}, double %{{.+}}, i1 %{{.*}})
__device__ void test_div_fmas_f64(double* out, double a, double b, double c, int d) {
  *out = __builtin_amdgcn_div_fmas(a, b, c, d);
}

// CIR-LABEL: @_Z15test_ds_swizzlePii
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.swizzle" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z15test_ds_swizzlePii
// LLVM: call{{.*}} i32 @llvm.amdgcn.ds.swizzle(i32 %{{.*}}, i32 32)
__device__ void test_ds_swizzle(int* out, int a) {
  *out = __builtin_amdgcn_ds_swizzle(a, 32);
}

// CIR-LABEL: @_Z18test_div_fixup_f32Pffff
// CIR: cir.call_llvm_intrinsic "amdgcn.div.fixup" {{.*}} : (!cir.float, !cir.float, !cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z18test_div_fixup_f32Pffff
// LLVM: call{{.*}} float @llvm.amdgcn.div.fixup.f32(float %{{.+}}, float %{{.+}}, float %{{.+}})
__device__ void test_div_fixup_f32(float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_div_fixupf(a, b, c);
}

// CIR-LABEL: @_Z18test_div_fixup_f64Pdddd
// CIR: cir.call_llvm_intrinsic "amdgcn.div.fixup" {{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z18test_div_fixup_f64Pdddd
// LLVM: call{{.*}} double @llvm.amdgcn.div.fixup.f64(double %{{.+}}, double %{{.+}}, double %{{.+}})
__device__ void test_div_fixup_f64(double* out, double a, double b, double c) {
  *out = __builtin_amdgcn_div_fixup(a, b, c);
}

// CIR-LABEL: @_Z13test_readlanePiii
// CIR: cir.call_llvm_intrinsic "amdgcn.readlane" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z13test_readlanePiii
// LLVM: call{{.*}} i32 @llvm.amdgcn.readlane.i32(i32 %{{.*}}, i32 %{{.*}})
__device__ void test_readlane(int* out, int a, int b) {
  *out = __builtin_amdgcn_readlane(a, b);
}

// CIR-LABEL: @_Z18test_readfirstlanePii
// CIR: cir.call_llvm_intrinsic "amdgcn.readfirstlane" {{.*}} : (!s32i) -> !s32i
// LLVM: define{{.*}} void @_Z18test_readfirstlanePii
// LLVM: call{{.*}} i32 @llvm.amdgcn.readfirstlane.i32(i32 %{{.*}})
__device__ void test_readfirstlane(int* out, int a) {
  *out = __builtin_amdgcn_readfirstlane(a);
}

// CIR-LABEL: @_Z17test_dispatch_ptr
// CIR: %{{.*}} = cir.call_llvm_intrinsic "amdgcn.dispatch.ptr" : () -> !cir.ptr<!void, target_address_space(4)>
// LLVM-LABEL: @_Z17test_dispatch_ptr
// LLVM: call{{.*}} ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
__device__ void test_dispatch_ptr(__attribute__((address_space(4))) void ** out) {
  *out = (__attribute__((address_space(4))) void *)__builtin_amdgcn_dispatch_ptr();
}

// CIR-LABEL: @_Z12test_rcp_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.rcp" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z12test_rcp_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.rcp.f32(float %{{.*}})
__device__ void test_rcp_f32(float* out, float a) {
  *out = __builtin_amdgcn_rcpf(a);
}

// CIR-LABEL: @_Z12test_rcp_f64Pdd
// CIR: cir.call_llvm_intrinsic "amdgcn.rcp" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z12test_rcp_f64Pdd
// LLVM: call{{.*}} double @llvm.amdgcn.rcp.f64(double %{{.*}})
__device__ void test_rcp_f64(double* out, double a) {
  *out = __builtin_amdgcn_rcp(a);
}

// CIR-LABEL: @_Z12test_rsq_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.rsq" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z12test_rsq_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.rsq.f32(float %{{.*}})
__device__ void test_rsq_f32(float* out, float a) {
  *out = __builtin_amdgcn_rsqf(a);
}

// CIR-LABEL: @_Z12test_rsq_f64Pdd
// CIR: cir.call_llvm_intrinsic "amdgcn.rsq" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z12test_rsq_f64Pdd
// LLVM: call{{.*}} double @llvm.amdgcn.rsq.f64(double %{{.*}})
__device__ void test_rsq_f64(double* out, double a) {
  *out = __builtin_amdgcn_rsq(a);
}

// CIR-LABEL: @_Z18test_rsq_clamp_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.rsq.clamp" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z18test_rsq_clamp_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.rsq.clamp.f32(float %{{.*}})
__device__ void test_rsq_clamp_f32(float* out, float a) {
  *out = __builtin_amdgcn_rsq_clampf(a);
}

// CIR-LABEL: @_Z18test_rsq_clamp_f64Pdd
// CIR: cir.call_llvm_intrinsic "amdgcn.rsq.clamp" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z18test_rsq_clamp_f64Pdd
// LLVM: call{{.*}} double @llvm.amdgcn.rsq.clamp.f64(double %{{.*}})
__device__ void test_rsq_clamp_f64(double* out, double a) {
  *out = __builtin_amdgcn_rsq_clamp(a);
}

// CIR-LABEL: @_Z13test_sqrt_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.sqrt" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z13test_sqrt_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.sqrt.f32(float %{{.*}})
__device__ void test_sqrt_f32(float* out, float a) {
  *out = __builtin_amdgcn_sqrtf(a);
}

// CIR-LABEL: @_Z13test_sqrt_f64Pdd
// CIR: cir.call_llvm_intrinsic "amdgcn.sqrt" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z13test_sqrt_f64Pdd
// LLVM: call{{.*}} double @llvm.amdgcn.sqrt.f64(double %{{.*}})
__device__ void test_sqrt_f64(double* out, double a) {
  *out = __builtin_amdgcn_sqrt(a);
}

// CIR-LABEL: @_Z13test_sinf_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.sin" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z13test_sinf_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.sin.f32(float %{{.*}})
__device__ void test_sinf_f32(float* out, float a) {
  *out = __builtin_amdgcn_sinf(a);
}

// CIR-LABEL: @_Z13test_cosf_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.cos" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z13test_cosf_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.cos.f32(float %{{.*}})
__device__ void test_cosf_f32(float* out, float a) {
  *out = __builtin_amdgcn_cosf(a);
}

// CIR-LABEL: @_Z13test_exp2_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.exp2" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z13test_exp2_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.exp2.f32(float %{{.*}})
__device__ void test_exp2_f32(float* out, float a) {
  *out = __builtin_amdgcn_exp2f(a);
}

// CIR-LABEL: @_Z12test_log_f32Pff
// CIR: cir.call_llvm_intrinsic "amdgcn.log" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z12test_log_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.log.f32(float %{{.*}})
__device__ void test_log_f32(float* out, float a) {
  *out = __builtin_amdgcn_logf(a);
}
