// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int2 ToTwoInts(int V) {
  return V.xx;
}

// CIR: %[[V_ADDR:.*]] = cir.alloca "V" {{.*}} init : !cir.ptr<!s32i>
// CIR: cir.store %[[ARG_V:.*]], %[[V_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_V:.*]] = cir.load {{.*}} %[[V_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<1 x !s32i>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CIR: %[[CAST_SPLAT:.*]] = cir.vec.insert %[[TMP_V]], %[[POISON]][%4 : !u64i] : !cir.vector<1 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<1 x !s32i>
// CIR: %[[RESULT:.*]] = cir.vec.shuffle(%[[CAST_SPLAT]], %[[POISON]] : !cir.vector<1 x !s32i>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !s32i>

// LLVM: %[[V_ADDR:.*]] = alloca i32, align 4
// LLVM: store i32 %[[ARG_V:.*]], ptr %[[V_ADDR]], align 4
// LLVM: %[[TMP_V:.*]] = load i32, ptr %[[V_ADDR]], align 4
// LLVM: %[[CAST_SPLAT:.*]] = insertelement <1 x i32> poison, i32 %[[TMP_V]], i64 0
// LLVM: %[[RESULT:.*]] = shufflevector <1 x i32> %[[CAST_SPLAT]], <1 x i32> poison, <2 x i32> zeroinitializer

float4 ToFourFloats(float V) {
  return V.rrrr;
}

// CIR: %[[V_ADDR:.*]] = cir.alloca "V" {{.*}} init : !cir.ptr<!cir.float>
// CIR: cir.store %[[ARG_V:.*]], %[[V_ADDR]] : !cir.float, !cir.ptr<!cir.float>
// CIR: %[[TMP_V:.*]] = cir.load {{.*}} %[[V_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<1 x !cir.float>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CIR: %[[CAST_SPLAT:.*]] = cir.vec.insert %[[TMP_V]], %[[POISON]][%[[ZERO]] : !u64i] : !cir.vector<1 x !cir.float>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<1 x !cir.float>
// CIR: %[[RESULT:.*]] = cir.vec.shuffle(%[[CAST_SPLAT]], %[[POISON]] : !cir.vector<1 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !cir.float>

// LLVM: %[[V_ADDR:.*]] = alloca float, align 4
// LLVM: store float %[[ARG_V:.*]], ptr %[[V_ADDR]], align 4
// LLVM: %[[TMP_V:.*]] = load float, ptr %[[V_ADDR]], align 4
// LLVM: %[[CAST_SPLAT:.*]] = insertelement <1 x float> poison, float %[[TMP_V]], i64 0
// LLVM: %[[RESULT:.*]] = shufflevector <1 x float> %[[CAST_SPLAT]], <1 x float> poison, <4 x i32> zeroinitializer

int2 FillOne() {
  return 1.xx;
}

// CIR: %[[TMP_ADDR:.*]] = cir.alloca "tmp" {{.*}} : !cir.ptr<!cir.vector<1 x !s32i>>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[CONST_1]] : !s32i, !cir.vector<1 x !s32i>
// CIR: cir.store {{.*}} %[[SPLAT]], %[[TMP_ADDR]] : !cir.vector<1 x !s32i>, !cir.ptr<!cir.vector<1 x !s32i>>
// CIR: %[[TMP:.*]] = cir.load {{.*}} %[[TMP_ADDR]] : !cir.ptr<!cir.vector<1 x !s32i>>, !cir.vector<1 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<1 x !s32i>
// CIR: %[[RESULT:.*]] = cir.vec.shuffle(%[[TMP]], %[[POISON]] : !cir.vector<1 x !s32i>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !s32i>

// LLVM: %[[TMP_ADDR:.*]] = alloca <1 x i32>, align 4
// LLVM: store <1 x i32> splat (i32 1), ptr %[[TMP_ADDR]], align 4
// LLVM: %[[TMP:.*]] = load <1 x i32>, ptr %[[TMP_ADDR]], align 4
// LLVM: %[[RESULT:.*]] = shufflevector <1 x i32> %[[TMP]], <1 x i32> poison, <2 x i32> zeroinitializer

uint3 FillOneUnsigned() {
  return 1u.xxx;
}

// CIR: %[[TMP_ADDR:.*]] = cir.alloca "tmp" {{.*}} : !cir.ptr<!cir.vector<1 x !u32i>>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !u32i
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[CONST_1]] : !u32i, !cir.vector<1 x !u32i>
// CIR: cir.store {{.*}} %[[SPLAT]], %[[TMP_ADDR]] : !cir.vector<1 x !u32i>, !cir.ptr<!cir.vector<1 x !u32i>>
// CIR: %[[TMP:.*]] = cir.load {{.*}} %[[TMP_ADDR]] : !cir.ptr<!cir.vector<1 x !u32i>>, !cir.vector<1 x !u32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<1 x !u32i>
// CIR: %[[RESULT:.*]] = cir.vec.shuffle(%[[TMP]], %[[POISON]] : !cir.vector<1 x !u32i>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<3 x !u32i>

// LLVM: %[[TMP_ADDR:.*]] = alloca <1 x i32>, align 4
// LLVM: store <1 x i32> splat (i32 1), ptr %[[TMP_ADDR]], align 4
// LLVM: %[[TMP:.*]] = load <1 x i32>, ptr %[[TMP_ADDR]], align 4
// LLVM: %[[RESULT:.*]] = shufflevector <1 x i32> %[[TMP]], <1 x i32> poison, <3 x i32> zeroinitializer

vector<uint64_t,4> FillOneUnsignedLong() {
  return 1ul.xxxx;
}

// CIR: %[[TMP_ADDR:.*]] = cir.alloca "tmp" {{.*}} : !cir.ptr<!cir.vector<1 x !u64i>>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !u64i
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[CONST_1]] : !u64i, !cir.vector<1 x !u64i>
// CIR: cir.store {{.*}} %[[SPLAT]], %[[TMP_ADDR]] : !cir.vector<1 x !u64i>, !cir.ptr<!cir.vector<1 x !u64i>>
// CIR: %[[TMP:.*]] = cir.load {{.*}} %[[TMP_ADDR]] : !cir.ptr<!cir.vector<1 x !u64i>>, !cir.vector<1 x !u64i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<1 x !u64i>
// CIR: %[[RESULT:.*]] = cir.vec.shuffle(%[[TMP]], %[[POISON]] : !cir.vector<1 x !u64i>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !u64i>

// LLVM: %[[TMP_ADDR:.*]] = alloca <1 x i64>, align 8
// LLVM: store <1 x i64> splat (i64 1), ptr %[[TMP_ADDR]], align 8
// LLVM: %[[TMP:.*]] = load <1 x i64>, ptr %[[TMP_ADDR]], align 8
// LLVM: %[[RESULT:.*]] = shufflevector <1 x i64> %[[TMP]], <1 x i64> poison, <4 x i32> zeroinitializer
