// RUN: %clang_cc1 %s -fclangir -emit-cir -triple spir-unknown-unknown -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR

// RUN: %clang_cc1 %s -fclangir -emit-llvm -triple spir-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

// RUN: %clang_cc1 %s -emit-llvm -triple spir-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

typedef __attribute__(( ext_vector_type(2) )) int int2;
typedef __attribute__(( ext_vector_type(4) )) int int4;
typedef __attribute__(( ext_vector_type(4) )) float float4;

int4 vec_ternary(int4 c, int4 a, int4 b) {
  return c ? a : c;
}

// CIR: %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
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

// CIR: %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.vector<4 x !cir.float>>
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

// CIR: %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.store %{{.*}}, %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[VEC_INC:.*]] = cir.inc %[[TMP_A]] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[VEC_INC]], %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[RESULT:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: cir.store %[[RESULT]], %[[RET_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_INC:.*]] = add <4 x i32> %[[A:.*]], splat (i32 1)
// LLVM: ret <4 x i32> %[[VEC_INC]]

int4 vec_unary_dec(int4 a) {
  --a;
  return a;
}

// CIR: %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.store %{{.*}}, %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[VEC_DEC:.*]] = cir.dec %[[TMP_A]] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[VEC_DEC]], %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[RESULT:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: cir.store %[[RESULT]], %[[RET_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_DEC:.*]] = add <4 x i32> %[[A:.*]], splat (i32 -1)
// LLVM: ret <4 x i32> %[[VEC_DEC]]

int4 vec_init_subvectors(int2 a, int2 b) {
  return (int4)(a, b);
}

// CIR-LABEL: cir.func {{.*}} @vec_init_subvectors
// CIR: %[[A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s32i>>, !cir.vector<2 x !s32i>
// CIR: %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s32i>>, !cir.vector<2 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[A_EXT:.*]] = cir.vec.shuffle(%[[A]], %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[V0:.*]] = cir.vec.shuffle(%[[A_EXT]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[B_EXT:.*]] = cir.vec.shuffle(%[[B]], %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.vec.shuffle(%[[V0]], %[[B_EXT]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !s32i>

// LLVM-LABEL: @vec_init_subvectors
// LLVM: %[[RESULT:.*]] = shufflevector <2 x i32> %{{.*}}, <2 x i32> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// LLVM: ret <4 x i32> %[[RESULT]]

int4 vec_init_scalar_and_subvector(int2 a, int c) {
  return (int4)(c, a, c);
}

// CIR-LABEL: cir.func {{.*}} @vec_init_scalar_and_subvector
// CIR: %[[C:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: %[[A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s32i>>, !cir.vector<2 x !s32i>
// CIR: %[[C2:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[V0:.*]] = cir.vec.insert %[[C]], %[[POISON]][%{{.*}} : !u64i] : !cir.vector<4 x !s32i>
// CIR: %[[A_EXT:.*]] = cir.vec.shuffle(%[[A]], %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[V1:.*]] = cir.vec.shuffle(%[[V0]], %[[A_EXT]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.vec.insert %[[C2]], %[[V1]][%{{.*}} : !u64i] : !cir.vector<4 x !s32i>

// LLVM-LABEL: @vec_init_scalar_and_subvector
// LLVM: %[[V0:.*]] = insertelement <4 x i32> poison, i32 %[[C:.*]], i64 0
// LLVM: %[[EXT:.*]] = shufflevector <2 x i32> %{{.*}}, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
// LLVM: %[[V1:.*]] = shufflevector <4 x i32> %[[V0]], <4 x i32> %[[EXT]], <4 x i32> <i32 0, i32 4, i32 5, i32 poison>
// LLVM: %[[V2:.*]] = insertelement <4 x i32> %[[V1]], i32 %[[C]], i64 3
// LLVM: ret <4 x i32> %[[V2]]

int4 vec_init_scalar_swizzles(int4 a, int c) {
  return (int4)(a.x, a.y, c, a.w);
}

// CIR-LABEL: cir.func {{.*}} @vec_init_scalar_swizzles
// CIR: %[[X:.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !s64i] : !cir.vector<4 x !s32i>
// CIR: %[[Y:.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !s64i] : !cir.vector<4 x !s32i>
// CIR: %[[C:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: %[[W:.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !s64i] : !cir.vector<4 x !s32i>
// CIR: cir.vec.create(%[[X]], %[[Y]], %[[C]], %[[W]] : !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>

// LLVM-LABEL: @vec_init_scalar_swizzles
// LLVM: %[[V0:.*]] = insertelement <4 x i32> %[[A:.*]], i32 %{{.*}}, i64 2
// LLVM: %[[RESULT:.*]] = shufflevector <4 x i32> %[[V0]], <4 x i32> %[[A]], <4 x i32> <i32 0, i32 1, i32 2, i32 7>
// LLVM: ret <4 x i32> %[[RESULT]]

int4 vec_init_subvector_swizzles(int4 a) {
  return (int4)(a.xy, a.zw);
}

// CIR-LABEL: cir.func {{.*}} @vec_init_subvector_swizzles
// CIR: %[[XY:.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !s32i>
// CIR: %[[ZW:.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[XY_EXT:.*]] = cir.vec.shuffle(%[[XY]], %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[V0:.*]] = cir.vec.shuffle(%[[XY_EXT]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[ZW_EXT:.*]] = cir.vec.shuffle(%[[ZW]], %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.vec.shuffle(%[[V0]], %[[ZW_EXT]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !s32i>

// LLVM-LABEL: @vec_init_subvector_swizzles
// LLVM-SAME: (<4 x i32> noundef returned %[[A:.*]])
// LLVM: ret <4 x i32> %[[A]]

int4 vec_init_scalar_and_swizzle(int4 a, int c) {
  return (int4)(c, a.yx, c);
}

// CIR-LABEL: cir.func {{.*}} @vec_init_scalar_and_swizzle
// CIR: %[[C:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: %[[YX:.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !s32i>
// CIR: %[[C2:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[V0:.*]] = cir.vec.insert %[[C]], %[[POISON]][%{{.*}} : !u64i] : !cir.vector<4 x !s32i>
// CIR: %[[YX_EXT:.*]] = cir.vec.shuffle(%[[YX]], %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[V1:.*]] = cir.vec.shuffle(%[[V0]], %[[YX_EXT]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.vec.insert %[[C2]], %[[V1]][%{{.*}} : !u64i] : !cir.vector<4 x !s32i>

// LLVM-LABEL: @vec_init_scalar_and_swizzle
// LLVM: %[[V0:.*]] = insertelement <4 x i32> poison, i32 %[[C:.*]], i64 0
// LLVM: %[[V1:.*]] = shufflevector <4 x i32> %[[V0]], <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 4, i32 poison>
// LLVM: %[[V2:.*]] = insertelement <4 x i32> %[[V1]], i32 %[[C]], i64 3
// LLVM: ret <4 x i32> %[[V2]]

int4 vec_init_narrow_swizzle(int2 b) {
  return (int4)(b.yx, b);
}

// CIR-LABEL: cir.func {{.*}} @vec_init_narrow_swizzle
// CIR: %[[YX:.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !s32i>
// CIR: %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s32i>>, !cir.vector<2 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[YX_EXT:.*]] = cir.vec.shuffle(%[[YX]], %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[V0:.*]] = cir.vec.shuffle(%[[YX_EXT]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[B_EXT:.*]] = cir.vec.shuffle(%[[B]], %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.vec.shuffle(%[[V0]], %[[B_EXT]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !s32i>

// LLVM-LABEL: @vec_init_narrow_swizzle
// LLVM: %[[RESULT:.*]] = shufflevector <2 x i32> %{{.*}}, <2 x i32> poison, <4 x i32> <i32 1, i32 0, i32 0, i32 1>
// LLVM: ret <4 x i32> %[[RESULT]]
