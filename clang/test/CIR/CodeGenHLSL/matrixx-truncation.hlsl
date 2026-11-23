// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void trunc_cast() {
  int4x4 a;
  int3x4 b = (int3x4) a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<4 x 4 x !s32i>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.matrix<3 x 4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<4 x 4 x !s32i>>, !cir.matrix<4 x 4 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.matrix<4 x 4 x !s32i>
// CIR: %[[RESULT:.*]] = cir.vec.shuffle(%[[TMP_A]], %[[POISON]] : !cir.matrix<4 x 4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i] : !cir.matrix<3 x 4 x !s32i> loc(#loc6)
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.matrix<3 x 4 x !s32i>, !cir.ptr<!cir.matrix<3 x 4 x !s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca [4 x <4 x i32>], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [4 x <3 x i32>], align 4
// LLVM: %[[TMP_A:.*]] = load <16 x i32>, ptr %[[A_ADDR]], align 4
// LLVM: %[[RESULT:.*]] = shufflevector <16 x i32> %[[TMP_A]], <16 x i32> poison, <12 x i32> <i32 0, i32 1, i32 2, i32 4, i32 5, i32 6, i32 8, i32 9, i32 10, i32 12, i32 13, i32 14>
// LLVM: store <12 x i32> %[[RESULT]], ptr %[[B_ADDR]], align 4
