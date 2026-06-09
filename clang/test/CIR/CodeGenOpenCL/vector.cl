// RUN: %clang_cc1 %s -fclangir -emit-cir -triple spir-unknown-unknown -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR

// RUN: %clang_cc1 %s -fclangir -emit-llvm -triple spir-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

// RUN: %clang_cc1 %s -emit-llvm -triple spir-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

typedef __attribute__(( ext_vector_type(4) )) int int4;
typedef __attribute__(( ext_vector_type(4) )) float float4;

int4 vec_ternary(int4 c, int4 a, int4 b) {
  return c ? a : c;
}

// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.store %{{.*}}, %[[C_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.store %{{.*}}, %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.store %{{.*}}, %[[B_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_C:.*]] = cir.load {{.*}} %[[C_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_C_2:.*]] = cir.load {{.*}} %[[C_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_0_VEC:.*]] = cir.const #cir.zero : !cir.vector<4 x !s32i>
// CIR: %[[C_CMP:.*]] = cir.vec.cmp(lt, %[[TMP_C]], %[[CONST_0_VEC]]) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
// CIR: %[[C_CMP_NOT:.*]] = cir.not %[[C_CMP]] : !cir.vector<4 x !s32i>
// CIR: %[[TMP_2:.*]] = cir.and %[[TMP_C_2]], %[[C_CMP_NOT]] : !cir.vector<4 x !s32i>
// CIR: %[[TMP_3:.*]] = cir.and %[[TMP_A]], %[[C_CMP]] : !cir.vector<4 x !s32i>
// CIR: %[[RESULT:.*]] = cir.or %[[TMP_2]], %[[TMP_3]] : !cir.vector<4 x !s32i>
// CIR: cir.store %[[RESULT]], %[[RET_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[RET_VAL:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: cir.return %[[RET_VAL]] : !cir.vector<4 x !s32i>

// LLVM: %[[COND_MAX:.*]] = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %[[COND:.*]], <4 x i32> zeroinitializer)
// LLVM: %[[COND_IS_NEG:.*]] = icmp slt <4 x i32> %[[COND]], zeroinitializer
// LLVM: %[[SELECT:.*]] = select <4 x i1> %[[COND_IS_NEG]], <4 x i32> %[[A:.*]], <4 x i32> zeroinitializer
// LLVM: %[[RESULT:.*]] = or <4 x i32> %[[SELECT]], %[[COND_MAX]]
// LLVM: ret <4 x i32> %[[RESULT]]

float4 vec_ternary_f4(int4 c, float4 a, float4 b) {
  return c ? a : b;
}

// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: cir.store %{{.*}}, %[[C_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.store %{{.*}}, %[[A_ADDR]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: cir.store %{{.*}}, %[[B_ADDR]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[TMP_C:.*]] = cir.load {{.*}} %[[C_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[CONST_0_VEC:.*]] = cir.const #cir.zero : !cir.vector<4 x !s32i>
// CIR: %[[C_CMP:.*]] = cir.vec.cmp(lt, %[[TMP_C]], %[[CONST_0_VEC]]) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
// CIR: %[[C_CMP_NOT:.*]] = cir.not %[[C_CMP]] : !cir.vector<4 x !s32i>
// CIR: %[[TMP_B_VI4:.*]] = cir.cast bitcast %[[TMP_B]] : !cir.vector<4 x !cir.float> -> !cir.vector<4 x !s32i>
// CIR: %[[TMP_A_VI4:.*]] = cir.cast bitcast %[[TMP_A]] : !cir.vector<4 x !cir.float> -> !cir.vector<4 x !s32i>
// CIR: %[[TMP_1:.*]] = cir.and %[[TMP_B_VI4]], %[[C_CMP_NOT]] : !cir.vector<4 x !s32i>
// CIR: %[[TMP_2:.*]] = cir.and %[[TMP_A_VI4]], %[[C_CMP]] : !cir.vector<4 x !s32i>
// CIR: %[[TMP_1_1_OR:.*]] = cir.or %[[TMP_1]], %[[TMP_2]] : !cir.vector<4 x !s32i>
// CIR: %[[RESULT:.*]] = cir.cast bitcast %[[TMP_1_1_OR]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>
// CIR: cir.store %[[RESULT]], %[[RET_ADDR]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[RET_VAL:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: cir.return %[[RET_VAL]] : !cir.vector<4 x !cir.float>

// LLVM: %[[IS_NEG:.*]] = icmp slt <4 x i32> %[[COND:.*]], zeroinitializer
// LLVM: %[[RESULT:.*]] = select <4 x i1> %[[IS_NEG]], <4 x float> %[[A:.*]], <4 x float> %[[B:.*]]
// LLVM: ret <4 x float> %[[RESULT]]

int4 vec_unary_inc(int4 a) {
  ++a;
  return a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[RET_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["__retval"]
// CIR: cir.store %{{.*}}, %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_1_VEC:.*]] = cir.const #cir.const_vector<[#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i]> : !cir.vector<4 x !s32i>
// CIR: %[[VEC_INC:.*]] = cir.add %[[TMP_A]], %[[CONST_1_VEC]] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[VEC_INC]], %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[RESULT:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: cir.store %[[RESULT]], %[[RET_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_INC:.*]] = add <4 x i32> %[[A:.*]], splat (i32 1)
// LLVM: ret <4 x i32> %[[VEC_INC]]

int4 vec_unary_dec(int4 a) {
  --a;
  return a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[RET_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["__retval"]
// CIR: cir.store %{{.*}}, %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_N1_VEC:.*]] = cir.const #cir.const_vector<[#cir.int<-1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i]> : !cir.vector<4 x !s32i>
// CIR: %[[VEC_DEC:.*]] = cir.add %[[TMP_A]], %[[CONST_N1_VEC]] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[VEC_DEC]], %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[RESULT:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: cir.store %[[RESULT]], %[[RET_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_DEC:.*]] = add <4 x i32> %[[A:.*]], splat (i32 -1)
// LLVM: ret <4 x i32> %[[VEC_DEC]]
