// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void use(const int *);
// CIR-DAG: cir.global "private" constant cir_private @[[CONST_ARR_16:.*]] = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 4> {alignment = 16 : i64}
// LLVM-DAG: @[[CONST_ARR_16:.*]] = private {{.*}}constant [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 16
// CIR-DAG: cir.global "private" constant cir_private @[[CONST_ARR_32:.*]] = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 4> {alignment = 32 : i64}
// LLVM-DAG: @[[CONST_ARR_32:.*]] = private {{.*}}constant [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 32

void f(bool condition) {
  if (condition) {
    alignas(16) int arr[4] = {1, 2, 3, 4};
    // CIR: %[[GET_GLOB:.*]] = cir.get_global @[[CONST_ARR_16]] : !cir.ptr<!cir.array<!s32i x 4>>
    // CIR: cir.copy %[[GET_GLOB]] align(16) to %{{.*}} align(16) : !cir.ptr<!cir.array<!s32i x 4>>
    // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.*}}, ptr align 16 @[[CONST_ARR_16]], i64 16, i1 false)
    use(arr);
  } else {
    alignas(32) int arr[4] = {1, 2, 3, 4};
    // CIR: %[[GET_GLOB:.*]] = cir.get_global @[[CONST_ARR_32]] : !cir.ptr<!cir.array<!s32i x 4>>
    // CIR: cir.copy %[[GET_GLOB]] align(32) to %{{.*}} align(32) : !cir.ptr<!cir.array<!s32i x 4>>
    // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 32 %{{.*}}, ptr align 32 @[[CONST_ARR_32]], i64 16, i1 false)
    use(arr);
  }
}
