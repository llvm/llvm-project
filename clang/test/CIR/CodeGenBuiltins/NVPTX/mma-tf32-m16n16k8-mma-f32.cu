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
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.mma.row.row.tf32"
// LLVM-LABEL: @test_0(
// LLVM-SAME: ptr noundef [[V0:%[-a-zA-Z$._0-9]+]], ptr noundef [[V1:%[-a-zA-Z$._0-9]+]], ptr noundef [[V2:%[-a-zA-Z$._0-9]+]], ptr noundef [[V3:%[-a-zA-Z$._0-9]+]])
// LLVM: store ptr [[V0]], ptr [[V4:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store ptr [[V1]], ptr [[V5:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store ptr [[V2]], ptr [[V6:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store ptr [[V3]], ptr [[V7:%[-a-zA-Z$._0-9]+]], align 8
// LLVM-DAG: [[V8:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V4]], align 8
// LLVM-DAG: [[V9:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V5]], align 8
// LLVM-DAG: [[V10:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V6]], align 8
// LLVM-DAG: [[V11:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V7]], align 8
// LLVM: [[V12:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 0
// LLVM: [[V13:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V12]], align 4
// LLVM: [[V14:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 1
// LLVM: [[V15:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V14]], align 4
// LLVM: [[V16:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 2
// LLVM: [[V17:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V16]], align 4
// LLVM: [[V18:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 3
// LLVM: [[V19:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V18]], align 4
// LLVM: [[V20:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 0
// LLVM: [[V21:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V20]], align 4
// LLVM: [[V22:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 1
// LLVM: [[V23:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V22]], align 4
// LLVM: [[V24:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 2
// LLVM: [[V25:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V24]], align 4
// LLVM: [[V26:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 3
// LLVM: [[V27:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V26]], align 4
// LLVM: [[V28:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 0
// LLVM: [[V29:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V28]], align 4
// LLVM: [[V30:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 1
// LLVM: [[V31:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V30]], align 4
// LLVM: [[V32:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 2
// LLVM: [[V33:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V32]], align 4
// LLVM: [[V34:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 3
// LLVM: [[V35:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V34]], align 4
// LLVM: [[V36:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 4
// LLVM: [[V37:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V36]], align 4
// LLVM: [[V38:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 5
// LLVM: [[V39:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V38]], align 4
// LLVM: [[V40:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 6
// LLVM: [[V41:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V40]], align 4
// LLVM: [[V42:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 7
// LLVM: [[V43:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V42]], align 4
// LLVM: [[V44:%[-a-zA-Z$._0-9]+]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.row.row.tf32(i32 [[V13]], i32 [[V15]], i32 [[V17]], i32 [[V19]], i32 [[V21]], i32 [[V23]], i32 [[V25]], i32 [[V27]], float [[V29]], float [[V31]], float [[V33]], float [[V35]], float [[V37]], float [[V39]], float [[V41]], float [[V43]])
// LLVM: [[V45:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V44]], 0
// LLVM: [[V46:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 0
// LLVM: store float [[V45]], ptr [[V46]], align 4
// LLVM: [[V47:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V44]], 1
// LLVM: [[V48:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 1
// LLVM: store float [[V47]], ptr [[V48]], align 4
// LLVM: [[V49:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V44]], 2
// LLVM: [[V50:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 2
// LLVM: store float [[V49]], ptr [[V50]], align 4
// LLVM: [[V51:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V44]], 3
// LLVM: [[V52:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 3
// LLVM: store float [[V51]], ptr [[V52]], align 4
// LLVM: [[V53:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V44]], 4
// LLVM: [[V54:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 4
// LLVM: store float [[V53]], ptr [[V54]], align 4
// LLVM: [[V55:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V44]], 5
// LLVM: [[V56:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 5
// LLVM: store float [[V55]], ptr [[V56]], align 4
// LLVM: [[V57:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V44]], 6
// LLVM: [[V58:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 6
// LLVM: store float [[V57]], ptr [[V58]], align 4
// LLVM: [[V59:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V44]], 7
// LLVM: [[V60:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 7
// LLVM: store float [[V59]], ptr [[V60]], align 4
// LLVM: ret void
extern "C" __attribute__((device)) void test_0(float *dst, const int *a, const int *b, const float *c) {
  __mma_tf32_m16n16k8_mma_f32(dst, a, b, c, 0, 0);
}

// CIR-LABEL: @test_1(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.mma.row.col.tf32"
// LLVM-LABEL: @test_1(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.row.col.tf32{{[^ (]*}}(i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
extern "C" __attribute__((device)) void test_1(float *dst, const int *a, const int *b, const float *c) {
  __mma_tf32_m16n16k8_mma_f32(dst, a, b, c, 1, 0);
}

// CIR-LABEL: @test_2(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.mma.col.row.tf32"
// LLVM-LABEL: @test_2(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.col.row.tf32{{[^ (]*}}(i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
extern "C" __attribute__((device)) void test_2(float *dst, const int *a, const int *b, const float *c) {
  __mma_tf32_m16n16k8_mma_f32(dst, a, b, c, 2, 0);
}

// CIR-LABEL: @test_3(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.mma.col.col.tf32"
// LLVM-LABEL: @test_3(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.col.col.tf32{{[^ (]*}}(i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
extern "C" __attribute__((device)) void test_3(float *dst, const int *a, const int *b, const float *c) {
  __mma_tf32_m16n16k8_mma_f32(dst, a, b, c, 3, 0);
}
