// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

void vector_truncation_to_scalar() {
  float4 a;
  float b = (float) a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.float>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !u64i
// CIR: %[[RESULT:.*]] = cir.vec.extract %[[TMP_A]][%[[CONST_0]] : !u64i] : !cir.vector<4 x !cir.float>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca <4 x float>, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, align 4
// LLVM: %[[TMP_A:.*]] = load <4 x float>, ptr %[[A_ADDR]], align 4
// LLVMCIR: %[[RESULT:.*]] = extractelement <4 x float> %[[TMP_A]], i64 0
// OGCG: %[[RESULT:.*]] = extractelement <4 x float> %[[TMP_A]], i32 0
// LLVM: store float %[[RESULT]], ptr %[[B_ADDR]], align 4

void vector_truncation_to_vector() {
  float4 a;
  float2 b = (float2) a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !cir.float>
// CIR: %[[RESULT:.*]] = cir.vec.shuffle(%[[TMP_A]], %[[POISON]] : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca <4 x float>, align 4
// LLVM: %[[B_ADDR:.*]] = alloca <2 x float>, align 4
// LLVM: %[[TMP_A:.*]] = load <4 x float>, ptr %[[A_ADDR]], align 4
// LLVM: %[[RESULT:.*]] = shufflevector <4 x float> %[[TMP_A]], <4 x float> poison, <2 x i32> <i32 0, i32 1>
// LLVM: store <2 x float> %[[RESULT]], ptr %[[B_ADDR]], align 4
