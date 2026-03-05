#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1010 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1011 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1012 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1010 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1011 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1012 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1010 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1011 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1012 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z15test_permlane16Pjjjjj
// CIR: cir.llvm.intrinsic "amdgcn.permlane16" {{.*}} : (!u32i, !u32i, !u32i, !u32i, !cir.bool, !cir.bool) -> !u32i
// LLVM: define{{.*}} void @_Z15test_permlane16Pjjjjj
// LLVM: call i32 @llvm.amdgcn.permlane16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 false)
// OGCG: define{{.*}} void @_Z15test_permlane16Pjjjjj
// OGCG: call i32 @llvm.amdgcn.permlane16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 false)
__device__ void test_permlane16(unsigned int* out, unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
  *out = __builtin_amdgcn_permlane16(a, b, c, d, 0, 0);
}


// CIR-LABEL: @_Z16test_permlanex16Pjjjjj
// CIR: cir.llvm.intrinsic "amdgcn.permlanex16" {{.*}} : (!u32i, !u32i, !u32i, !u32i, !cir.bool, !cir.bool) -> !u32i
// LLVM: define{{.*}} void @_Z16test_permlanex16Pjjjjj
// LLVM: call i32 @llvm.amdgcn.permlanex16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 false)
// OGCG: define{{.*}} void @_Z16test_permlanex16Pjjjjj
// OGCG: call i32 @llvm.amdgcn.permlanex16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 false)
__device__ void test_permlanex16(unsigned int* out, unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 0, 0);
}
