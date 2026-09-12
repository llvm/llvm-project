// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu9.42-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.42-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.42-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR
// calls yet, so LLVM check lines use {{.*}} to tolerate the difference
// between CIR (no flags) and classic codegen ('contract').

#define __device__ __attribute__((device))
#define global __attribute__((address_space(1)))

typedef _Float16 v2f16 __attribute__((ext_vector_type(2)));

// CIR-LABEL: @_Z19test_atomic_add_i32u22__amdgpu_buffer_rsrc_tiii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.atomic.add" {{.*}} -> !s32i
// LLVM: define{{.*}} i32 @_Z19test_atomic_add_i32u22__amdgpu_buffer_rsrc_tiii
// LLVM: call{{.*}} i32 @llvm.amdgcn.raw.ptr.buffer.atomic.add.i32(i32 {{.*}})
__device__ int test_atomic_add_i32(__amdgpu_buffer_rsrc_t rsrc, int x, int offset,
                                  int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @_Z20test_atomic_fadd_f32u22__amdgpu_buffer_rsrc_tfii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.atomic.fadd" {{.*}} -> !cir.float
// LLVM: define{{.*}} float @_Z20test_atomic_fadd_f32u22__amdgpu_buffer_rsrc_tfii
// LLVM: call{{.*}} float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(float {{.*}})
__device__ float test_atomic_fadd_f32(__amdgpu_buffer_rsrc_t rsrc, float x, int offset,
                                     int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @_Z20test_atomic_fadd_f64u22__amdgpu_buffer_rsrc_tdii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.atomic.fadd" {{.*}} -> !cir.double
// LLVM: define{{.*}} double @_Z20test_atomic_fadd_f64u22__amdgpu_buffer_rsrc_tdii
// LLVM: call{{.*}} double @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f64(double {{.*}})
__device__ double test_atomic_fadd_f64(__amdgpu_buffer_rsrc_t rsrc, double x, int offset,
                                      int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f64(x, rsrc, offset, soffset, 0);
}

// CIR-LABEL: @_Z22test_atomic_fadd_v2f16u22__amdgpu_buffer_rsrc_tDv2_DF16_ii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.atomic.fadd" {{.*}} -> !cir.vector<2 x !cir.f16>
// LLVM: define{{.*}} <2 x half> @_Z22test_atomic_fadd_v2f16u22__amdgpu_buffer_rsrc_tDv2_DF16_ii
// LLVM: call{{.*}} <2 x half> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2f16(<2 x half> {{.*}})
__device__ v2f16 test_atomic_fadd_v2f16(__amdgpu_buffer_rsrc_t rsrc, v2f16 x, int offset,
                                       int soffset) {
  return __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(x, rsrc, offset, soffset, 0);
}

