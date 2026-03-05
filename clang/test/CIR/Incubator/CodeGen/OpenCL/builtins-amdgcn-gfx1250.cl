// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1250 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1250 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1250 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CIR-LABEL: @test_rcp_bf16
// CIR: cir.llvm.intrinsic "amdgcn.rcp" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @test_rcp_bf16
// LLVM: call{{.*}} bfloat @llvm.amdgcn.rcp.bf16(bfloat %{{.*}})
// OGCG: define{{.*}} void @test_rcp_bf16
// OGCG: call{{.*}} bfloat @llvm.amdgcn.rcp.bf16(bfloat %{{.*}})
void test_rcp_bf16(global __bf16* out, __bf16 a)
{
  *out = __builtin_amdgcn_rcp_bf16(a);
}

// CIR-LABEL: @test_sqrt_bf16
// CIR: cir.llvm.intrinsic "amdgcn.sqrt" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @test_sqrt_bf16
// LLVM: call{{.*}} bfloat @llvm.amdgcn.sqrt.bf16(bfloat %{{.*}})
// OGCG: define{{.*}} void @test_sqrt_bf16
// OGCG: call{{.*}} bfloat @llvm.amdgcn.sqrt.bf16(bfloat %{{.*}})
void test_sqrt_bf16(global __bf16* out, __bf16 a)
{
  *out = __builtin_amdgcn_sqrt_bf16(a);
}

// CIR-LABEL: @test_rsq_bf16
// CIR: cir.llvm.intrinsic "amdgcn.rsq" {{.*}} : (!cir.bf16) -> !cir.bf16
// LLVM: define{{.*}} void @test_rsq_bf16
// LLVM: call{{.*}} bfloat @llvm.amdgcn.rsq.bf16(bfloat %{{.*}})
// OGCG: define{{.*}} void @test_rsq_bf16
// OGCG: call{{.*}} bfloat @llvm.amdgcn.rsq.bf16(bfloat %{{.*}})
void test_rsq_bf16(__bf16* out, __bf16 a)
{
  *out = __builtin_amdgcn_rsq_bf16(a);
}
