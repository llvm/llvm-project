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
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.store.d.row.stride.f32"
// LLVM-LABEL: @test_0(
// LLVM-SAME: ptr noundef [[V0:%[-a-zA-Z$._0-9]+]], ptr noundef [[V1:%[-a-zA-Z$._0-9]+]], i32 noundef [[V2:%[-a-zA-Z$._0-9]+]])
// LLVM: store ptr [[V0]], ptr [[V3:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store ptr [[V1]], ptr [[V4:%[-a-zA-Z$._0-9]+]], align 8
// LLVM: store i32 [[V2]], ptr [[V5:%[-a-zA-Z$._0-9]+]], align 4
// LLVM-DAG: [[V6:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V3]], align 8
// LLVM-DAG: [[V7:%[-a-zA-Z$._0-9]+]] = load ptr, ptr [[V4]], align 8
// LLVM-DAG: [[V8:%[-a-zA-Z$._0-9]+]] = load i32, ptr [[V5]], align 4
// LLVM: [[V9:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V7]], i{{32|64}} 0
// LLVM: [[V10:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V9]], align 4
// LLVM: [[V11:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V7]], i{{32|64}} 1
// LLVM: [[V12:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V11]], align 4
// LLVM: [[V13:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V7]], i{{32|64}} 2
// LLVM: [[V14:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V13]], align 4
// LLVM: [[V15:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V7]], i{{32|64}} 3
// LLVM: [[V16:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V15]], align 4
// LLVM: [[V17:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V7]], i{{32|64}} 4
// LLVM: [[V18:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V17]], align 4
// LLVM: [[V19:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V7]], i{{32|64}} 5
// LLVM: [[V20:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V19]], align 4
// LLVM: [[V21:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V7]], i{{32|64}} 6
// LLVM: [[V22:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V21]], align 4
// LLVM: [[V23:%[-a-zA-Z$._0-9]+]] = getelementptr float, ptr [[V7]], i{{32|64}} 7
// LLVM: [[V24:%[-a-zA-Z$._0-9]+]] = load float, ptr [[V23]], align 4
// LLVM: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(ptr [[V6]], float [[V10]], float [[V12]], float [[V14]], float [[V16]], float [[V18]], float [[V20]], float [[V22]], float [[V24]], i32 [[V8]])
// LLVM: ret void
extern "C" __attribute__((device)) void test_0(float *dst, const float *src, unsigned ldm) {
  __hmma_m16n16k16_st_c_f32(dst, src, ldm, 0);
}

// CIR-LABEL: @test_1(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.store.d.col.stride.f32"
// LLVM-LABEL: @test_1(
// LLVM: call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
extern "C" __attribute__((device)) void test_1(float *dst, const float *src, unsigned ldm) {
  __hmma_m16n16k16_st_c_f32(dst, src, ldm, 1);
}

// CIR-LABEL: @test_2(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.store.d.col.stride.f32"
// LLVM-LABEL: @test_2(
// LLVM: call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
extern "C" __attribute__((device)) void test_2(float *dst, const float *src, unsigned ldm) {
  __hmma_m16n16k16_st_c_f32(dst, src, ldm, -1);
}

// CIR-LABEL: @test_3(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.store.d.col.stride.f32"
// LLVM-LABEL: @test_3(
// LLVM: call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32{{[^ (]*}}(ptr {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, float {{%[-a-zA-Z$._0-9]+}}, i32 {{%[-a-zA-Z$._0-9]+}})
extern "C" __attribute__((device)) void test_3(float *dst, const float *src, unsigned ldm) {
  __hmma_m16n16k16_st_c_f32(dst, src, ldm, 7);
}
