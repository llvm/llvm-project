// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu13.10-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu13.10-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu13.10-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR
// calls yet, so LLVM check lines use {{.*}} to tolerate the difference
// between CIR (no flags) and classic codegen ('contract').

#define __device__ __attribute__((device))
#define global __attribute__((address_space(1)))

typedef _Float16 v2f16 __attribute__((ext_vector_type(2)));

// CIR-LABEL: @_Z36test_raw_ptr_buffer_atomic_fmin_fmaxPU3AS1fPU3AS1du22__amdgpu_buffer_rsrc_tfdii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.atomic.fmin" {{.*}} -> !cir.float
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.atomic.fmax" {{.*}} -> !cir.float
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.atomic.fmin" {{.*}} -> !cir.double
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.atomic.fmax" {{.*}} -> !cir.double
// LLVM: define{{.*}} void @_Z36test_raw_ptr_buffer_atomic_fmin_fmaxPU3AS1fPU3AS1du22__amdgpu_buffer_rsrc_tfdii
// LLVM: call{{.*}} float @llvm.amdgcn.raw.ptr.buffer.atomic.fmin.f32(float {{.*}})
// LLVM: call{{.*}} float @llvm.amdgcn.raw.ptr.buffer.atomic.fmax.f32(float {{.*}})
// LLVM: call{{.*}} double @llvm.amdgcn.raw.ptr.buffer.atomic.fmin.f64(double {{.*}})
// LLVM: call{{.*}} double @llvm.amdgcn.raw.ptr.buffer.atomic.fmax.f64(double {{.*}})
__device__ void test_raw_ptr_buffer_atomic_fmin_fmax(global float *fout,
                                                   global double *dout,
                                                   __amdgpu_buffer_rsrc_t rsrc,
                                                   float f, double d, int offset,
                                                   int soffset) {
  *fout = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32(f, rsrc, offset, soffset, 0);
  *fout = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(f, rsrc, offset, soffset, 0);
  *dout = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64(d, rsrc, offset, soffset, 0);
  *dout = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(d, rsrc, offset, soffset, 0);
}
