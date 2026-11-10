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
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.row.stride.f16"
// LLVM-LABEL: @test_0(
// LLVM-SAME: ptr noundef [[V0:%[-a-zA-Z$._0-9]+]], ptr noundef [[V1:%[-a-zA-Z$._0-9]+]], i32 noundef [[V2:%[-a-zA-Z$._0-9]+]])
// LLVM: store ptr [[V0]], ptr [[V3:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store ptr [[V1]], ptr [[V4:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store i32 [[V2]], ptr [[V5:%[-a-zA-Z$._0-9]+]], align 4
// LLVM-DAG: [[V6:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V3]], align 8
// LLVM-DAG: [[V7:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V4]], align 8
// LLVM-DAG: [[V8:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V5]], align 4
// LLVM: [[V9:%[-a-zA-Z$._0-9]+]] = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr [[V7]], i32 [[V8]])
// LLVM: [[V10:%[-a-zA-Z$._0-9]+]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[V9]], 0
// LLVM: [[V11:%[-a-zA-Z$._0-9]+]] = bitcast <2 x half> [[V10]] to i32
// LLVM: [[V12:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 0
// LLVM: store i32 [[V11]], ptr [[V12]], align 4
// LLVM: [[V13:%[-a-zA-Z$._0-9]+]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[V9]], 1
// LLVM: [[V14:%[-a-zA-Z$._0-9]+]] = bitcast <2 x half> [[V13]] to i32
// LLVM: [[V15:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 1
// LLVM: store i32 [[V14]], ptr [[V15]], align 4
// LLVM: [[V16:%[-a-zA-Z$._0-9]+]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[V9]], 2
// LLVM: [[V17:%[-a-zA-Z$._0-9]+]] = bitcast <2 x half> [[V16]] to i32
// LLVM: [[V18:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 2
// LLVM: store i32 [[V17]], ptr [[V18]], align 4
// LLVM: [[V19:%[-a-zA-Z$._0-9]+]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[V9]], 3
// LLVM: [[V20:%[-a-zA-Z$._0-9]+]] = bitcast <2 x half> [[V19]] to i32
// LLVM: [[V21:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 3
// LLVM: store i32 [[V20]], ptr [[V21]], align 4
// LLVM: [[V22:%[-a-zA-Z$._0-9]+]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[V9]], 4
// LLVM: [[V23:%[-a-zA-Z$._0-9]+]] = bitcast <2 x half> [[V22]] to i32
// LLVM: [[V24:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 4
// LLVM: store i32 [[V23]], ptr [[V24]], align 4
// LLVM: [[V25:%[-a-zA-Z$._0-9]+]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[V9]], 5
// LLVM: [[V26:%[-a-zA-Z$._0-9]+]] = bitcast <2 x half> [[V25]] to i32
// LLVM: [[V27:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 5
// LLVM: store i32 [[V26]], ptr [[V27]], align 4
// LLVM: [[V28:%[-a-zA-Z$._0-9]+]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[V9]], 6
// LLVM: [[V29:%[-a-zA-Z$._0-9]+]] = bitcast <2 x half> [[V28]] to i32
// LLVM: [[V30:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 6
// LLVM: store i32 [[V29]], ptr [[V30]], align 4
// LLVM: [[V31:%[-a-zA-Z$._0-9]+]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[V9]], 7
// LLVM: [[V32:%[-a-zA-Z$._0-9]+]] = bitcast <2 x half> [[V31]] to i32
// LLVM: [[V33:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V6]], i{{32|64}} 7
// LLVM: store i32 [[V32]], ptr [[V33]], align 4
// LLVM: ret void
extern "C" __attribute__((device)) void test_0(int *dst, const int *src, unsigned ldm) {
  __hmma_m16n16k16_ld_a(dst, src, ldm, 0);
}

// CIR-LABEL: @test_1(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.f16"
// LLVM-LABEL: @test_1(
// LLVM: [[R:%.*]] = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 0
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 1
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 2
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 3
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 4
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 5
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 6
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 7
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_1(int *dst, const int *src, unsigned ldm) {
  __hmma_m16n16k16_ld_a(dst, src, ldm, 1);
}

// CIR-LABEL: @test_2(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.f16"
// LLVM-LABEL: @test_2(
// LLVM: [[R:%.*]] = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 0
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 1
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 2
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 3
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 4
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 5
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 6
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 7
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_2(int *dst, const int *src, unsigned ldm) {
  __hmma_m16n16k16_ld_a(dst, src, ldm, -1);
}

// CIR-LABEL: @test_3(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.f16"
// LLVM-LABEL: @test_3(
// LLVM: [[R:%.*]] = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 0
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 1
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 2
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 3
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 4
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 5
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 6
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM: extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } [[R]], 7
// LLVM: bitcast <2 x half> {{%.*}} to i32
// LLVM: store i32 {{.*}}, ptr {{.*}}, align 4
extern "C" __attribute__((device)) void test_3(int *dst, const int *src, unsigned ldm) {
  __hmma_m16n16k16_ld_a(dst, src, ldm, 7);
}
