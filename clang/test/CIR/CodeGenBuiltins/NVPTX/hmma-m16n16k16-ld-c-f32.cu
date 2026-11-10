// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -target-feature +ptx70 -fcuda-is-device -ffp-contract=off -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -target-feature +ptx70 -fcuda-is-device -ffp-contract=off -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.cir.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -target-feature +ptx70 -fcuda-is-device -ffp-contract=off -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Classic counterpart: clang/test/CodeGen/builtins-nvptx-mma.cu,
// test_wmma_buitins. The first case also checks fragment data flow.

// CIR-LABEL: @test_0(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.row.stride.f32"
// LLVM-LABEL: @test_0(
// LLVM-SAME: ptr noundef [[V0:%[-a-zA-Z$._0-9]+]], ptr noundef [[V1:%[-a-zA-Z$._0-9]+]], i32 noundef [[V2:%[-a-zA-Z$._0-9]+]])
// LLVM: store ptr [[V0]], ptr [[V3:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store ptr [[V1]], ptr [[V4:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store i32 [[V2]], ptr [[V5:%[-a-zA-Z$._0-9]+]], align 4
// LLVM-DAG: [[V6:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V3]], align 8
// LLVM-DAG: [[V7:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V4]], align 8
// LLVM-DAG: [[V8:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V5]], align 4
// LLVM: [[V9:%[-a-zA-Z$._0-9]+]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p0(ptr [[V7]], i32 [[V8]])
// LLVM: [[V10:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V9]], 0
// LLVM: [[V11:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V6]], i{{32|64}} 0
// LLVM: store float [[V10]], ptr [[V11]], align 4
// LLVM: [[V12:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V9]], 1
// LLVM: [[V13:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V6]], i{{32|64}} 1
// LLVM: store float [[V12]], ptr [[V13]], align 4
// LLVM: [[V14:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V9]], 2
// LLVM: [[V15:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V6]], i{{32|64}} 2
// LLVM: store float [[V14]], ptr [[V15]], align 4
// LLVM: [[V16:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V9]], 3
// LLVM: [[V17:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V6]], i{{32|64}} 3
// LLVM: store float [[V16]], ptr [[V17]], align 4
// LLVM: [[V18:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V9]], 4
// LLVM: [[V19:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V6]], i{{32|64}} 4
// LLVM: store float [[V18]], ptr [[V19]], align 4
// LLVM: [[V20:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V9]], 5
// LLVM: [[V21:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V6]], i{{32|64}} 5
// LLVM: store float [[V20]], ptr [[V21]], align 4
// LLVM: [[V22:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V9]], 6
// LLVM: [[V23:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V6]], i{{32|64}} 6
// LLVM: store float [[V22]], ptr [[V23]], align 4
// LLVM: [[V24:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V9]], 7
// LLVM: [[V25:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V6]], i{{32|64}} 7
// LLVM: store float [[V24]], ptr [[V25]], align 4
// LLVM: ret void
extern "C" __attribute__((device)) void test_0(float *dst, const float *src, unsigned ldm) {
  __hmma_m16n16k16_ld_c_f32(dst, src, ldm, 0);
}

// CIR-LABEL: @test_1(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.col.stride.f32"
// LLVM-LABEL: @test_1(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: [[E0:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 0
// LLVM: store float [[E0]], ptr {{.*}}, align 4
// LLVM: [[E1:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 1
// LLVM: store float [[E1]], ptr {{.*}}, align 4
// LLVM: [[E2:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 2
// LLVM: store float [[E2]], ptr {{.*}}, align 4
// LLVM: [[E3:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 3
// LLVM: store float [[E3]], ptr {{.*}}, align 4
// LLVM: [[E4:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 4
// LLVM: store float [[E4]], ptr {{.*}}, align 4
// LLVM: [[E5:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 5
// LLVM: store float [[E5]], ptr {{.*}}, align 4
// LLVM: [[E6:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 6
// LLVM: store float [[E6]], ptr {{.*}}, align 4
// LLVM: [[E7:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 7
// LLVM: store float [[E7]], ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_1(float *dst, const float *src, unsigned ldm) {
  __hmma_m16n16k16_ld_c_f32(dst, src, ldm, 1);
}

// CIR-LABEL: @test_2(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.col.stride.f32"
// LLVM-LABEL: @test_2(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: [[E0:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 0
// LLVM: store float [[E0]], ptr {{.*}}, align 4
// LLVM: [[E1:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 1
// LLVM: store float [[E1]], ptr {{.*}}, align 4
// LLVM: [[E2:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 2
// LLVM: store float [[E2]], ptr {{.*}}, align 4
// LLVM: [[E3:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 3
// LLVM: store float [[E3]], ptr {{.*}}, align 4
// LLVM: [[E4:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 4
// LLVM: store float [[E4]], ptr {{.*}}, align 4
// LLVM: [[E5:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 5
// LLVM: store float [[E5]], ptr {{.*}}, align 4
// LLVM: [[E6:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 6
// LLVM: store float [[E6]], ptr {{.*}}, align 4
// LLVM: [[E7:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 7
// LLVM: store float [[E7]], ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_2(float *dst, const float *src, unsigned ldm) {
  __hmma_m16n16k16_ld_c_f32(dst, src, ldm, -1);
}

// CIR-LABEL: @test_3(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.col.stride.f32"
// LLVM-LABEL: @test_3(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: [[E0:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 0
// LLVM: store float [[E0]], ptr {{.*}}, align 4
// LLVM: [[E1:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 1
// LLVM: store float [[E1]], ptr {{.*}}, align 4
// LLVM: [[E2:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 2
// LLVM: store float [[E2]], ptr {{.*}}, align 4
// LLVM: [[E3:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 3
// LLVM: store float [[E3]], ptr {{.*}}, align 4
// LLVM: [[E4:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 4
// LLVM: store float [[E4]], ptr {{.*}}, align 4
// LLVM: [[E5:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 5
// LLVM: store float [[E5]], ptr {{.*}}, align 4
// LLVM: [[E6:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 6
// LLVM: store float [[E6]], ptr {{.*}}, align 4
// LLVM: [[E7:%.*]] = extractvalue { float, float, float, float, float, float, float, float } [[R]], 7
// LLVM: store float [[E7]], ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_3(float *dst, const float *src, unsigned ldm) {
  __hmma_m16n16k16_ld_c_f32(dst, src, ldm, 7);
}
