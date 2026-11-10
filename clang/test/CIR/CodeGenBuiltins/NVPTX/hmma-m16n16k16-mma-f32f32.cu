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
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f32.f32"
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
// LLVM: [[V14:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V13]] to <2 x half>
// LLVM: [[V15:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 1
// LLVM: [[V16:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V15]], align 4
// LLVM: [[V17:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V16]] to <2 x half>
// LLVM: [[V18:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 2
// LLVM: [[V19:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V18]], align 4
// LLVM: [[V20:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V19]] to <2 x half>
// LLVM: [[V21:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 3
// LLVM: [[V22:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V21]], align 4
// LLVM: [[V23:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V22]] to <2 x half>
// LLVM: [[V24:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 4
// LLVM: [[V25:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V24]], align 4
// LLVM: [[V26:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V25]] to <2 x half>
// LLVM: [[V27:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 5
// LLVM: [[V28:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V27]], align 4
// LLVM: [[V29:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V28]] to <2 x half>
// LLVM: [[V30:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 6
// LLVM: [[V31:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V30]], align 4
// LLVM: [[V32:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V31]] to <2 x half>
// LLVM: [[V33:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V9]], i{{32|64}} 7
// LLVM: [[V34:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V33]], align 4
// LLVM: [[V35:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V34]] to <2 x half>
// LLVM: [[V36:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 0
// LLVM: [[V37:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V36]], align 4
// LLVM: [[V38:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V37]] to <2 x half>
// LLVM: [[V39:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 1
// LLVM: [[V40:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V39]], align 4
// LLVM: [[V41:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V40]] to <2 x half>
// LLVM: [[V42:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 2
// LLVM: [[V43:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V42]], align 4
// LLVM: [[V44:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V43]] to <2 x half>
// LLVM: [[V45:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 3
// LLVM: [[V46:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V45]], align 4
// LLVM: [[V47:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V46]] to <2 x half>
// LLVM: [[V48:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 4
// LLVM: [[V49:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V48]], align 4
// LLVM: [[V50:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V49]] to <2 x half>
// LLVM: [[V51:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 5
// LLVM: [[V52:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V51]], align 4
// LLVM: [[V53:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V52]] to <2 x half>
// LLVM: [[V54:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 6
// LLVM: [[V55:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V54]], align 4
// LLVM: [[V56:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V55]] to <2 x half>
// LLVM: [[V57:%[-a-zA-Z$._0-9]+]] = getelementptr i32, ptr [[V10]], i{{32|64}} 7
// LLVM: [[V58:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V57]], align 4
// LLVM: [[V59:%[-a-zA-Z$._0-9]+]] = bitcast i32 [[V58]] to <2 x half>
// LLVM: [[V60:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 0
// LLVM: [[V61:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V60]], align 4
// LLVM: [[V62:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 1
// LLVM: [[V63:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V62]], align 4
// LLVM: [[V64:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 2
// LLVM: [[V65:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V64]], align 4
// LLVM: [[V66:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 3
// LLVM: [[V67:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V66]], align 4
// LLVM: [[V68:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 4
// LLVM: [[V69:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V68]], align 4
// LLVM: [[V70:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 5
// LLVM: [[V71:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V70]], align 4
// LLVM: [[V72:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 6
// LLVM: [[V73:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V72]], align 4
// LLVM: [[V74:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V11]], i{{32|64}} 7
// LLVM: [[V75:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V74]], align 4
// LLVM: [[V76:%[-a-zA-Z$._0-9]+]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[V14]], <2 x half> [[V17]], <2 x half> [[V20]], <2 x half> [[V23]], <2 x half> [[V26]], <2 x half> [[V29]], <2 x half> [[V32]], <2 x half> [[V35]], <2 x half> [[V38]], <2 x half> [[V41]], <2 x half> [[V44]], <2 x half> [[V47]], <2 x half> [[V50]], <2 x half> [[V53]], <2 x half> [[V56]], <2 x half> [[V59]], float [[V61]], float [[V63]], float [[V65]], float [[V67]], float [[V69]], float [[V71]], float [[V73]], float [[V75]])
// LLVM: [[V77:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V76]], 0
// LLVM: [[V78:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 0
// LLVM: store float [[V77]], ptr [[V78]], align 4
// LLVM: [[V79:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V76]], 1
// LLVM: [[V80:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 1
// LLVM: store float [[V79]], ptr [[V80]], align 4
// LLVM: [[V81:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V76]], 2
// LLVM: [[V82:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 2
// LLVM: store float [[V81]], ptr [[V82]], align 4
// LLVM: [[V83:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V76]], 3
// LLVM: [[V84:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 3
// LLVM: store float [[V83]], ptr [[V84]], align 4
// LLVM: [[V85:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V76]], 4
// LLVM: [[V86:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 4
// LLVM: store float [[V85]], ptr [[V86]], align 4
// LLVM: [[V87:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V76]], 5
// LLVM: [[V88:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 5
// LLVM: store float [[V87]], ptr [[V88]], align 4
// LLVM: [[V89:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V76]], 6
// LLVM: [[V90:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 6
// LLVM: store float [[V89]], ptr [[V90]], align 4
// LLVM: [[V91:%[-a-zA-Z$._0-9]+]] = extractvalue { float, float, float, float, float, float, float, float } [[V76]], 7
// LLVM: [[V92:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V8]], i{{32|64}} 7
// LLVM: store float [[V91]], ptr [[V92]], align 4
// LLVM: ret void
extern "C" __attribute__((device)) void test_0(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 0, 0);
}

// CIR-LABEL: @test_1(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f32.f32"
// LLVM-LABEL: @test_1(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32{{[^ (]*}}(<2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 1, 0);
}

// CIR-LABEL: @test_2(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f32.f32"
// LLVM-LABEL: @test_2(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f32{{[^ (]*}}(<2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 2, 0);
}

// CIR-LABEL: @test_3(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f32.f32"
// LLVM-LABEL: @test_3(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32{{[^ (]*}}(<2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 3, 0);
}

// CIR-LABEL: @test_4(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f32.f32.satfinite"
// LLVM-LABEL: @test_4(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32.satfinite{{[^ (]*}}(<2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
extern "C" __attribute__((device)) void test_4(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 0, 1);
}

// CIR-LABEL: @test_5(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f32.f32.satfinite"
// LLVM-LABEL: @test_5(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32.satfinite{{[^ (]*}}(<2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
extern "C" __attribute__((device)) void test_5(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 1, 1);
}

// CIR-LABEL: @test_6(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f32.f32.satfinite"
// LLVM-LABEL: @test_6(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f32.satfinite{{[^ (]*}}(<2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
extern "C" __attribute__((device)) void test_6(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 2, 1);
}

// CIR-LABEL: @test_7(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f32.f32.satfinite"
// LLVM-LABEL: @test_7(
// LLVM: [[R:%.*]] = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32.satfinite{{[^ (]*}}(<2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, <2 x half> {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}})
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
extern "C" __attribute__((device)) void test_7(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 3, 1);
}
