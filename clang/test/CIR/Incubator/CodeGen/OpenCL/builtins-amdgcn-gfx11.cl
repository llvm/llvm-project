// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1100 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1101 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1102 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1103 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1150 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1151 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1152 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1153 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1100 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1101 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1102 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1103 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1150 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1151 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1152 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1153 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1100 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1101 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1102 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1103 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1150 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1151 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1152 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1153 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef unsigned int uint;
typedef unsigned long ulong;

// CIR-LABEL: @test_permlanex16
// CIR: cir.llvm.intrinsic "amdgcn.permlanex16" {{.*}} : (!u32i, !u32i, !u32i, !u32i, !cir.bool, !cir.bool) -> !u32i
// LLVM: define{{.*}} void @test_permlanex16
// LLVM: call i32 @llvm.amdgcn.permlanex16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 false)
// OGCG: define{{.*}} void @test_permlanex16
// OGCG: call i32 @llvm.amdgcn.permlanex16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 false)
void test_permlanex16(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 0, 0);
}
