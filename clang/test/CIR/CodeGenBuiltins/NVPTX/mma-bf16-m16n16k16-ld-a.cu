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
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.row.stride.bf16"
// LLVM-LABEL: @test_0(
// LLVM-SAME: ptr noundef [[V0:%[-a-zA-Z$._0-9]+]], ptr noundef [[V1:%[-a-zA-Z$._0-9]+]], i32 noundef [[V2:%[-a-zA-Z$._0-9]+]])
// LLVM: store ptr [[V0]], ptr [[V3:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store ptr [[V1]], ptr [[V4:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store i32 [[V2]], ptr [[V5:%[-a-zA-Z$._0-9]+]], align 4
// LLVM-DAG: [[V6:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V3]], align 8
// LLVM-DAG: [[V7:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V4]], align 8
// LLVM-DAG: [[V8:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V5]], align 4
// LLVM: [[V9:%[-a-zA-Z$._0-9]+]] = call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.bf16.p0(ptr [[V7]], i32 [[V8]])
// LLVM: [[V10:%[-a-zA-Z$._0-9]+]] = extractvalue { i32, i32, i32, i32 } [[V9]], 0
// LLVM: [[V11:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 0
// LLVM: store i32 [[V10]], ptr [[V11]], align 4
// LLVM: [[V12:%[-a-zA-Z$._0-9]+]] = extractvalue { i32, i32, i32, i32 } [[V9]], 1
// LLVM: [[V13:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 1
// LLVM: store i32 [[V12]], ptr [[V13]], align 4
// LLVM: [[V14:%[-a-zA-Z$._0-9]+]] = extractvalue { i32, i32, i32, i32 } [[V9]], 2
// LLVM: [[V15:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 2
// LLVM: store i32 [[V14]], ptr [[V15]], align 4
// LLVM: [[V16:%[-a-zA-Z$._0-9]+]] = extractvalue { i32, i32, i32, i32 } [[V9]], 3
// LLVM: [[V17:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 3
// LLVM: store i32 [[V16]], ptr [[V17]], align 4
// LLVM: ret void
extern "C" __attribute__((device)) void test_0(int *dst, const int *src, unsigned ldm) {
  __mma_bf16_m16n16k16_ld_a(dst, src, ldm, 0);
}

// CIR-LABEL: @test_1(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.bf16"
// LLVM-LABEL: @test_1(
// LLVM: [[R:%.*]] = call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.bf16{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 0
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 1
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 2
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 3
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_1(int *dst, const int *src, unsigned ldm) {
  __mma_bf16_m16n16k16_ld_a(dst, src, ldm, 1);
}

// CIR-LABEL: @test_2(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.bf16"
// LLVM-LABEL: @test_2(
// LLVM: [[R:%.*]] = call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.bf16{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 0
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 1
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 2
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 3
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_2(int *dst, const int *src, unsigned ldm) {
  __mma_bf16_m16n16k16_ld_a(dst, src, ldm, -1);
}

// CIR-LABEL: @test_3(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.bf16"
// LLVM-LABEL: @test_3(
// LLVM: [[R:%.*]] = call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.bf16{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 0
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 1
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 2
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { i32, i32, i32, i32 } [[R]], 3
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_3(int *dst, const int *src, unsigned ldm) {
  __mma_bf16_m16n16k16_ld_a(dst, src, ldm, 7);
}
