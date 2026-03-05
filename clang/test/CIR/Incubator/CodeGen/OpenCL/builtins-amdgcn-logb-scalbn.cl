// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1100 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1100 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1100 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test logb/logbf and scalbn/scalbnf builtins
//===----------------------------------------------------------------------===//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// CIR-LABEL: @test_logbf
// CIR: cir.call @logbf({{.*}}) : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @test_logbf(
// LLVM: call {{.*}}float @logbf(float %{{.*}})
// OGCG: define{{.*}} void @test_logbf(
// OGCG: call { float, i32 } @llvm.frexp.f32.i32(float %{{.*}})
// OGCG: extractvalue { float, i32 } %{{.*}}, 1
// OGCG: add nsw i32 %{{.*}}, -1
// OGCG: sitofp i32 %{{.*}} to float
// OGCG: call {{.*}}float @llvm.fabs.f32(float %{{.*}})
// OGCG: fcmp {{.*}}one float %{{.*}}, 0x7FF0000000000000
// OGCG: select {{.*}}i1 %{{.*}}, float %{{.*}}, float %{{.*}}
// OGCG: fcmp {{.*}}oeq float %{{.*}}, 0.000000e+00
// OGCG: select {{.*}}i1 %{{.*}}, float 0xFFF0000000000000, float %{{.*}}
void test_logbf(global float* out, float a) {
  *out = __builtin_logbf(a);
}

// CIR-LABEL: @test_logb
// CIR: cir.call @logb({{.*}}) : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @test_logb(
// LLVM: call {{.*}}double @logb(double %{{.*}})
// OGCG: define{{.*}} void @test_logb(
// OGCG: call { double, i32 } @llvm.frexp.f64.i32(double %{{.*}})
// OGCG: extractvalue { double, i32 } %{{.*}}, 1
// OGCG: add nsw i32 %{{.*}}, -1
// OGCG: sitofp i32 %{{.*}} to double
// OGCG: call {{.*}}double @llvm.fabs.f64(double %{{.*}})
// OGCG: fcmp {{.*}}one double %{{.*}}, 0x7FF0000000000000
// OGCG: select {{.*}}i1 %{{.*}}, double %{{.*}}, double %{{.*}}
// OGCG: fcmp {{.*}}oeq double %{{.*}}, 0.000000e+00
// OGCG: select {{.*}}i1 %{{.*}}, double 0xFFF0000000000000, double %{{.*}}
void test_logb(global double* out, double a) {
  *out = __builtin_logb(a);
}

// CIR-LABEL: @test_scalbnf
// CIR: cir.call @scalbnf({{.*}}) : (!cir.float, !s32i) -> !cir.float
// LLVM: define{{.*}} void @test_scalbnf(
// LLVM: call {{.*}}float @scalbnf(float %{{.*}}, i32 %{{.*}})
// OGCG: define{{.*}} void @test_scalbnf(
// OGCG: call {{.*}}float @llvm.ldexp.f32.i32(float %{{.*}}, i32 %{{.*}})
void test_scalbnf(global float* out, float a, int b) {
  *out = __builtin_scalbnf(a, b);
}

// CIR-LABEL: @test_scalbn
// CIR: cir.call @scalbn({{.*}}) : (!cir.double, !s32i) -> !cir.double
// LLVM: define{{.*}} void @test_scalbn(
// LLVM: call {{.*}}double @scalbn(double %{{.*}}, i32 %{{.*}})
// OGCG: define{{.*}} void @test_scalbn(
// OGCG: call {{.*}}double @llvm.ldexp.f64.i32(double %{{.*}}, i32 %{{.*}})
void test_scalbn(global double* out, double a, int b) {
  *out = __builtin_scalbn(a, b);
}
