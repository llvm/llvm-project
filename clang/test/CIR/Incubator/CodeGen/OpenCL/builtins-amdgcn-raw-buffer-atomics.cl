// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx90a -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx90a -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx90a \
// RUN:            -target-feature +atomic-fmin-fmax-global-f32 \
// RUN:            -target-feature +atomic-fmin-fmax-global-f64 \
// RUN:            -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test raw buffer atomic builtins
//===----------------------------------------------------------------------===//

typedef half __attribute__((ext_vector_type(2))) float16x2_t;

// CIR-LABEL: @test_atomic_add_i32
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.atomic.add" {{.*}} : (!s32i, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !s32i
// LLVM-LABEL: define{{.*}} i32 @test_atomic_add_i32
// LLVM: call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.add.i32(i32 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
// OGCG-LABEL: define{{.*}} i32 @test_atomic_add_i32
// OGCG: call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.add.i32(i32 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
int test_atomic_add_i32(__amdgpu_buffer_rsrc_t rsrc, int x, int offset, int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @test_atomic_fadd_f32
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.atomic.fadd" {{.*}} : (!cir.float, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.float
// LLVM-LABEL: define{{.*}} float @test_atomic_fadd_f32
// LLVM: call float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(float %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
// OGCG-LABEL: define{{.*}} float @test_atomic_fadd_f32
// OGCG: call float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(float %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
float test_atomic_fadd_f32(__amdgpu_buffer_rsrc_t rsrc, float x, int offset, int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @test_atomic_fadd_v2f16
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.atomic.fadd" {{.*}} : (!cir.vector<!cir.f16 x 2>, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.vector<!cir.f16 x 2>
// LLVM-LABEL: define{{.*}} <2 x half> @test_atomic_fadd_v2f16
// LLVM: call <2 x half> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2f16(<2 x half> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
// OGCG-LABEL: define{{.*}} <2 x half> @test_atomic_fadd_v2f16
// OGCG: call <2 x half> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2f16(<2 x half> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
float16x2_t test_atomic_fadd_v2f16(__amdgpu_buffer_rsrc_t rsrc, float16x2_t x, int offset, int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @test_atomic_fmin_f32
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.atomic.fmin" {{.*}} : (!cir.float, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.float
// LLVM-LABEL: define{{.*}} float @test_atomic_fmin_f32
// LLVM: call float @llvm.amdgcn.raw.ptr.buffer.atomic.fmin.f32(float %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
// OGCG-LABEL: define{{.*}} float @test_atomic_fmin_f32
// OGCG: call float @llvm.amdgcn.raw.ptr.buffer.atomic.fmin.f32(float %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
float test_atomic_fmin_f32(__amdgpu_buffer_rsrc_t rsrc, float x, int offset, int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @test_atomic_fmin_f64
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.atomic.fmin" {{.*}} : (!cir.double, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.double
// LLVM-LABEL: define{{.*}} double @test_atomic_fmin_f64
// LLVM: call double @llvm.amdgcn.raw.ptr.buffer.atomic.fmin.f64(double %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
// OGCG-LABEL: define{{.*}} double @test_atomic_fmin_f64
// OGCG: call double @llvm.amdgcn.raw.ptr.buffer.atomic.fmin.f64(double %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
double test_atomic_fmin_f64(__amdgpu_buffer_rsrc_t rsrc, double x, int offset, int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @test_atomic_fmax_f32
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.atomic.fmax" {{.*}} : (!cir.float, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.float
// LLVM-LABEL: define{{.*}} float @test_atomic_fmax_f32
// LLVM: call float @llvm.amdgcn.raw.ptr.buffer.atomic.fmax.f32(float %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
// OGCG-LABEL: define{{.*}} float @test_atomic_fmax_f32
// OGCG: call float @llvm.amdgcn.raw.ptr.buffer.atomic.fmax.f32(float %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
float test_atomic_fmax_f32(__amdgpu_buffer_rsrc_t rsrc, float x, int offset, int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @test_atomic_fmax_f64
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.atomic.fmax" {{.*}} : (!cir.double, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.double
// LLVM-LABEL: define{{.*}} double @test_atomic_fmax_f64
// LLVM: call double @llvm.amdgcn.raw.ptr.buffer.atomic.fmax.f64(double %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
// OGCG-LABEL: define{{.*}} double @test_atomic_fmax_f64
// OGCG: call double @llvm.amdgcn.raw.ptr.buffer.atomic.fmax.f64(double %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
double test_atomic_fmax_f64(__amdgpu_buffer_rsrc_t rsrc, double x, int offset, int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(x, rsrc, offset, soffset, 0);
}
