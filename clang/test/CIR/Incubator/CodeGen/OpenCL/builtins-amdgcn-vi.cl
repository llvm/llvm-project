// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu tonga -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx900 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1010 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1012 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu tonga -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx900 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1010 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1012 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu tonga -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx900 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1010 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1012 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CIR-LABEL: @test_div_fixup_f16
// CIR: cir.llvm.intrinsic "amdgcn.div.fixup" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @test_div_fixup_f16
// LLVM: call{{.*}} half @llvm.amdgcn.div.fixup.f16(half %{{.+}}, half %{{.+}}, half %{{.+}})
// OGCG: define{{.*}} void @test_div_fixup_f16
// OGCG: call{{.*}} half @llvm.amdgcn.div.fixup.f16(half %{{.+}}, half %{{.+}}, half %{{.+}})
void test_div_fixup_f16(global half* out, half a, half b, half c) {
  *out = __builtin_amdgcn_div_fixuph(a, b, c);
}

// CIR-LABEL: @test_rcp_f16
// CIR: cir.llvm.intrinsic "amdgcn.rcp" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @test_rcp_f16
// LLVM: call{{.*}} half @llvm.amdgcn.rcp.f16(half %{{.*}})
// OGCG: define{{.*}} void @test_rcp_f16
// OGCG: call{{.*}} half @llvm.amdgcn.rcp.f16(half %{{.*}})
void test_rcp_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_rcph(a);
}

// CIR-LABEL: @test_sqrt_f16
// CIR: cir.llvm.intrinsic "amdgcn.sqrt" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @test_sqrt_f16
// LLVM: call{{.*}} half @llvm.{{((amdgcn.){0,1})}}sqrt.f16(half %{{.*}})
// OGCG: define{{.*}} void @test_sqrt_f16
// OGCG: call{{.*}} half @llvm.{{((amdgcn.){0,1})}}sqrt.f16(half %{{.*}})
void test_sqrt_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_sqrth(a);
}

// CIR-LABEL: @test_rsq_f16
// CIR: cir.llvm.intrinsic "amdgcn.rsq" {{.*}} : (!cir.f16) -> !cir.f16
// LLVM: define{{.*}} void @test_rsq_f16
// LLVM: call{{.*}} half @llvm.amdgcn.rsq.f16(half %{{.*}})
// OGCG: define{{.*}} void @test_rsq_f16
// OGCG: call{{.*}} half @llvm.amdgcn.rsq.f16(half %{{.*}})
void test_rsq_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_rsqh(a);
}
