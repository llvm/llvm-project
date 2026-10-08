// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#define __device__ __attribute__((device))
#define global __attribute__((address_space(1)))

// CIR-LABEL: @_Z15test_bitop3_b32PU3AS1jjjj
// CIR: cir.call_llvm_intrinsic "amdgcn.bitop3" {{.*}} : (!s32i, !s32i, !s32i, !u32i) -> !s32i
// LLVM: define{{.*}} void @_Z15test_bitop3_b32PU3AS1jjjj
// LLVM: call{{.*}} i32 @llvm.amdgcn.bitop3.i32(i32 %{{.+}}, i32 %{{.+}}, i32 %{{.+}}, i32 1)
__device__ void test_bitop3_b32(global unsigned int* out, unsigned int a,
                                unsigned int b, unsigned int c) {
  *out = __builtin_amdgcn_bitop3_b32(a, b, c, 1);
}

// CIR-LABEL: @_Z15test_bitop3_b16PU3AS1tttt
// CIR: cir.call_llvm_intrinsic "amdgcn.bitop3" {{.*}} : (!s16i, !s16i, !s16i, !u32i) -> !s16i
// LLVM: define{{.*}} void @_Z15test_bitop3_b16PU3AS1tttt
// LLVM: call{{.*}} i16 @llvm.amdgcn.bitop3.i16(i16 %{{.+}}, i16 %{{.+}}, i16 %{{.+}}, i32 1)
__device__ void test_bitop3_b16(global unsigned short* out, unsigned short a,
                                unsigned short b, unsigned short c) {
  *out = __builtin_amdgcn_bitop3_b16(a, b, c, 1);
}
