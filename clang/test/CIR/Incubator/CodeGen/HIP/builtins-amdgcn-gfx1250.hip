#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1250 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1250 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1250 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z13test_rcp_bf16PDF16bDF16b
// CIR: cir.llvm.intrinsic "amdgcn.rcp" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z13test_rcp_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.rcp.bf16(bfloat %{{.*}})
// OGCG: define{{.*}} void @_Z13test_rcp_bf16PDF16bDF16b
// OGCG: call{{.*}} bfloat @llvm.amdgcn.rcp.bf16(bfloat %{{.*}})
__device__ void test_rcp_bf16(__bf16* out, __bf16 a)
{
  *out = __builtin_amdgcn_rcp_bf16(a);
}

// CIR-LABEL: @_Z14test_sqrt_bf16PDF16bDF16b
// CIR: cir.llvm.intrinsic "amdgcn.sqrt" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z14test_sqrt_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.sqrt.bf16(bfloat %{{.*}})
// OGCG: define{{.*}} void @_Z14test_sqrt_bf16PDF16bDF16b
// OGCG: call{{.*}} bfloat @llvm.amdgcn.sqrt.bf16(bfloat %{{.*}})
__device__ void test_sqrt_bf16(__bf16* out, __bf16 a)
{
  *out = __builtin_amdgcn_sqrt_bf16(a);
}

// CIR-LABEL: @_Z13test_rsq_bf16PDF16bDF16b
// CIR: cir.llvm.intrinsic "amdgcn.rsq" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @_Z13test_rsq_bf16PDF16bDF16b
// LLVM: call{{.*}} bfloat @llvm.amdgcn.rsq.bf16(bfloat %{{.*}})
// OGCG: define{{.*}} void @_Z13test_rsq_bf16PDF16bDF16b
// OGCG: call{{.*}} bfloat @llvm.amdgcn.rsq.bf16(bfloat %{{.*}})
__device__ void test_rsq_bf16(__bf16* out, __bf16 a)
{
  *out = __builtin_amdgcn_rsq_bf16(a);
}
