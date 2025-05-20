// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef int vi4 __attribute__((ext_vector_type(4)));
typedef int vi3 __attribute__((ext_vector_type(3)));
typedef int vi2 __attribute__((ext_vector_type(2)));
typedef double vd2 __attribute__((ext_vector_type(2)));

vi4 vec_a;
// CIR: cir.global external @[[VEC_A:.*]] = #cir.zero : !cir.vector<4 x !s32i>

// LLVM: @[[VEC_A:.*]] = dso_local global <4 x i32> zeroinitializer

// OGCG: @[[VEC_A:.*]] = global <4 x i32> zeroinitializer

vi3 vec_b;
// CIR: cir.global external @[[VEC_B:.*]] = #cir.zero : !cir.vector<3 x !s32i>

// LLVM: @[[VEC_B:.*]] = dso_local global <3 x i32> zeroinitializer

// OGCG: @[[VEC_B:.*]] = global <3 x i32> zeroinitializer

vi2 vec_c;
// CIR: cir.global external @[[VEC_C:.*]] = #cir.zero : !cir.vector<2 x !s32i>

// LLVM: @[[VEC_C:.*]] = dso_local global <2 x i32> zeroinitializer

// OGCG: @[[VEC_C:.*]] = global <2 x i32> zeroinitializer

vd2 vec_d;

// CIR: cir.global external @[[VEC_D:.*]] = #cir.zero : !cir.vector<2 x !cir.double>

// LLVM: @[[VEC_D:.*]] = dso_local global <2 x double> zeroinitialize

// OGCG: @[[VEC_D:.*]] = global <2 x double> zeroinitializer

vi4 vec_e = { 1, 2, 3, 4 };

// CIR: cir.global external @[[VEC_E:.*]] = #cir.const_vector<[#cir.int<1> : !s32i, #cir.int<2> :
// CIR-SAME: !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.vector<4 x !s32i>

// LLVM: @[[VEC_E:.*]] = dso_local global <4 x i32> <i32 1, i32 2, i32 3, i32 4>

// OGCG: @[[VEC_E:.*]] = global <4 x i32> <i32 1, i32 2, i32 3, i32 4>

int x = 5;

void foo() {
  vi4 a;
  vi3 b;
  vi2 c;
  vd2 d;

  vi4 e = { 1, 2, 3, 4 };

  vi4 f = { x, 5, 6, x + 1 };

  vi4 g = { 5 };

  vi4 h = {};
}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[VEC_B:.*]] = cir.alloca !cir.vector<3 x !s32i>, !cir.ptr<!cir.vector<3 x !s32i>>, ["b"]
// CIR: %[[VEC_C:.*]] = cir.alloca !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>, ["c"]
// CIR: %[[VEC_D:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["d"]
// CIR: %[[VEC_E:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["e", init]
// CIR: %[[VEC_F:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["f", init]
// CIR: %[[VEC_G:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["g", init]
// CIR: %[[VEC_H:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["h", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_E_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_E_VAL]], %[[VEC_E]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[GLOBAL_X:.*]] = cir.get_global @x : !cir.ptr<!s32i>
// CIR: %[[X_VAL:.*]] = cir.load %[[GLOBAL_X]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[CONST_5:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[CONST_6:.*]] = cir.const #cir.int<6> : !s32i
// CIR: %[[GLOBAL_X:.*]] = cir.get_global @x : !cir.ptr<!s32i>
// CIR: %[[X:.*]] = cir.load %[[GLOBAL_X]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[X_PLUS_1:.*]] = cir.binop(add, %[[X]], %[[CONST_1]]) nsw : !s32i
// CIR: %[[VEC_F_VAL:.*]] = cir.vec.create(%[[X_VAL]], %[[CONST_5]], %[[CONST_6]], %[[X_PLUS_1]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_F_VAL]], %[[VEC_F]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_5:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[VEC_G_VAL:.*]] = cir.vec.create(%[[CONST_5]], %[[CONST_0]], %[[CONST_0]], %[[CONST_0]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_G_VAL]], %[[VEC_G]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[VEC_H_VAL:.*]] = cir.vec.create(%[[ZERO]], %[[ZERO]], %[[ZERO]], %[[ZERO]] : !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_H_VAL]], %[[VEC_H]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_B:.*]] = alloca <3 x i32>, i64 1, align 16
// LLVM: %[[VEC_C:.*]] = alloca <2 x i32>, i64 1, align 8
// LLVM: %[[VEC_D:.*]] = alloca <2 x double>, i64 1, align 16
// LLVM: %[[VEC_E:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_F:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_G:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_H:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_E]], align 16
// LLVM: store <4 x i32> {{.*}}, ptr %[[VEC_F]], align 16
// LLVM: store <4 x i32> <i32 5, i32 0, i32 0, i32 0>, ptr %[[VEC_G]], align 16
// LLVM: store <4 x i32> zeroinitializer, ptr %[[VEC_H]], align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_B:.*]] = alloca <3 x i32>, align 16
// OGCG: %[[VEC_C:.*]] = alloca <2 x i32>, align 8
// OGCG: %[[VEC_D:.*]] = alloca <2 x double>, align 16
// OGCG: %[[VEC_E:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_F:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_G:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_H:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_E]], align 16
// OGCG: store <4 x i32> {{.*}}, ptr %[[VEC_F]], align 16
// OGCG: store <4 x i32> <i32 5, i32 0, i32 0, i32 0>, ptr %[[VEC_G]], align 16
// OGCG: store <4 x i32> zeroinitializer, ptr %[[VEC_H]], align 16

void foo2(vi4 p) {}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["p", init]
// CIR: cir.store %{{.*}}, %[[VEC_A]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> %{{.*}}, ptr %[[VEC_A]], align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> %{{.*}}, ptr %[[VEC_A]], align 16

void foo3() {
  vi4 a = { 1, 2, 3, 4 };
  int e = a[1];
}

// CIR: %[[VEC:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_VAL]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP:.*]] = cir.load %[[VEC]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[IDX:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ELE:.*]] = cir.vec.extract %[[TMP]][%[[IDX]] : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store %[[ELE]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[VEC:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// LLVM: %[[TMP:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: %[[ELE:.*]] = extractelement <4 x i32> %[[TMP]], i32 1
// LLVM: store i32 %[[ELE]], ptr %[[INIT]], align 4

// OGCG: %[[VEC:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// OGCG: %[[TMP:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: %[[ELE:.*]] = extractelement <4 x i32> %[[TMP]], i32 1
// OGCG: store i32 %[[ELE]], ptr %[[INIT]], align 4

void foo4() {
  vi4 a = { 1, 2, 3, 4 };

  int idx = 2;
  int e = a[idx];
}

// CIR: %[[VEC:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[IDX:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["idx", init]
// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_VAL]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_IDX:.*]] = cir.const #cir.int<2> : !s32i
// CIR: cir.store %[[CONST_IDX]], %[[IDX]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP1:.*]] = cir.load %[[VEC]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP2:.*]] = cir.load %[[IDX]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[ELE:.*]] = cir.vec.extract %[[TMP1]][%[[TMP2]] : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store %[[ELE]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[VEC:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[IDX:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// LLVM: store i32 2, ptr %[[IDX]], align 4
// LLVM: %[[TMP1:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: %[[TMP2:.*]] = load i32, ptr %[[IDX]], align 4
// LLVM: %[[ELE:.*]] = extractelement <4 x i32> %[[TMP1]], i32 %[[TMP2]]
// LLVM: store i32 %[[ELE]], ptr %[[INIT]], align 4

// OGCG: %[[VEC:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[IDX:.*]] = alloca i32, align 4
// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// OGCG: store i32 2, ptr %[[IDX]], align 4
// OGCG: %[[TMP1:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: %[[TMP2:.*]] = load i32, ptr %[[IDX]], align 4
// OGCG: %[[ELE:.*]] = extractelement <4 x i32> %[[TMP1]], i32 %[[TMP2]]
// OGCG: store i32 %[[ELE]], ptr %[[INIT]], align 4

void foo5() {
  vi4 a = { 1, 2, 3, 4 };

  a[2] = 5;
}

// CIR: %[[VEC:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_VAL]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_VAL:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[CONST_IDX:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[TMP:.*]] = cir.load %[[VEC]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[NEW_VEC:.*]] = cir.vec.insert %[[CONST_VAL]], %[[TMP]][%[[CONST_IDX]] : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store %[[NEW_VEC]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// LLVM: %[[TMP:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: %[[NEW_VEC:.*]] = insertelement <4 x i32> %[[TMP]], i32 5, i32 2
// LLVM: store <4 x i32> %[[NEW_VEC]], ptr %[[VEC]], align 16

// OGCG: %[[VEC:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// OGCG: %[[TMP:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: %[[NEW_VEC:.*]] = insertelement <4 x i32> %[[TMP]], i32 5, i32 2
// OGCG: store <4 x i32> %[[NEW_VEC]], ptr %[[VEC]], align 16

void foo6() {
  vi4 a = { 1, 2, 3, 4 };
  int idx = 2;
  int value = 5;
  a[idx] = value;
}

// CIR: %[[VEC:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[IDX:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["idx", init]
// CIR: %[[VAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_VAL]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_IDX:.*]] = cir.const #cir.int<2> : !s32i
// CIR: cir.store %[[CONST_IDX]], %[[IDX]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[CONST_VAL:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store %[[CONST_VAL]], %[[VAL]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP1:.*]] = cir.load %[[VAL]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[TMP2:.*]] = cir.load %[[IDX]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[TMP3:.*]] = cir.load %0 : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[NEW_VEC:.*]] = cir.vec.insert %[[TMP1]], %[[TMP3]][%[[TMP2]] : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store %[[NEW_VEC]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[IDX:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[VAL:.*]] = alloca i32, i64 1, align 4
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %1, align 16
// LLVM: store i32 2, ptr %[[IDX]], align 4
// LLVM: store i32 5, ptr %[[VAL]], align 4
// LLVM: %[[TMP1:.*]] = load i32, ptr %[[VAL]], align 4
// LLVM: %[[TMP2:.*]] = load i32, ptr %[[IDX]], align 4
// LLVM: %[[TMP3:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: %[[NEW_VEC:.*]] = insertelement <4 x i32> %[[TMP3]], i32 %[[TMP1]], i32 %[[TMP2]]
// LLVM: store <4 x i32> %[[NEW_VEC]], ptr %[[VEC]], align 16

// OGCG: %[[VEC:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[IDX:.*]] = alloca i32, align 4
// OGCG: %[[VAL:.*]] = alloca i32, align 4
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// OGCG: store i32 2, ptr %[[IDX]], align 4
// OGCG: store i32 5, ptr %[[VAL]], align 4
// OGCG: %[[TMP1:.*]] = load i32, ptr %[[VAL]], align 4
// OGCG: %[[TMP2:.*]] = load i32, ptr %[[IDX]], align 4
// OGCG: %[[TMP3:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: %[[NEW_VEC:.*]] = insertelement <4 x i32> %[[TMP3]], i32 %[[TMP1]], i32 %[[TMP2]]
// OGCG: store <4 x i32> %[[NEW_VEC]], ptr %[[VEC]], align 16

void foo7() {
  vi4 a = {1, 2, 3, 4};
  a[2] += 5;
}

// CIR: %[[VEC:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_VAL]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_VAL:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[CONST_IDX:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[TMP:.*]] = cir.load %[[VEC]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[ELE:.*]] = cir.vec.extract %[[TMP]][%[[CONST_IDX]] : !s32i] : !cir.vector<4 x !s32i>
// CIR: %[[RES:.*]] = cir.binop(add, %[[ELE]], %[[CONST_VAL]]) nsw : !s32i
// CIR: %[[TMP2:.*]] = cir.load %[[VEC]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[NEW_VEC:.*]] = cir.vec.insert %[[RES]], %[[TMP2]][%[[CONST_IDX]] : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store %[[NEW_VEC]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// LLVM: %[[TMP:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: %[[ELE:.*]] = extractelement <4 x i32> %[[TMP]], i32 2
// LLVM: %[[RES:.*]] = add nsw i32 %[[ELE]], 5
// LLVM: %[[TMP2:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: %[[NEW_VEC:.*]] = insertelement <4 x i32> %[[TMP2]], i32 %[[RES]], i32 2
// LLVM: store <4 x i32> %[[NEW_VEC]], ptr %[[VEC]], align 16

// OGCG: %[[VEC:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// OGCG: %[[TMP:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: %[[ELE:.*]] = extractelement <4 x i32> %[[TMP]], i32 2
// OGCG: %[[RES:.*]] = add nsw i32 %[[ELE]], 5
// OGCG: %[[TMP2:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: %[[NEW_VEC:.*]] = insertelement <4 x i32> %[[TMP2]], i32 %[[RES]], i32 2
// OGCG: store <4 x i32> %[[NEW_VEC]], ptr %[[VEC]], align 16


void foo8() {
  vi4 a = { 1, 2, 3, 4 };
  vi4 plus_res = +a;
  vi4 minus_res = -a;
  vi4 not_res = ~a;
}

// CIR: %[[VEC:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[PLUS_RES:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["plus_res", init]
// CIR: %[[MINUS_RES:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["minus_res", init]
// CIR: %[[NOT_RES:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["not_res", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_VAL]], %[[VEC]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP1:.*]] = cir.load %[[VEC]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[PLUS:.*]] = cir.unary(plus, %[[TMP1]]) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
// CIR: cir.store %[[PLUS]], %[[PLUS_RES]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP2:.*]] = cir.load %[[VEC]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[MINUS:.*]] = cir.unary(minus, %[[TMP2]]) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
// CIR: cir.store %[[MINUS]], %[[MINUS_RES]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP3:.*]] = cir.load %[[VEC]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[NOT:.*]] = cir.unary(not, %[[TMP3]]) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
// CIR: cir.store %[[NOT]], %[[NOT_RES]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[PLUS_RES:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[MINUS_RES:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[NOT_RES:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// LLVM: %[[TMP1:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: store <4 x i32> %[[TMP1]], ptr %[[PLUS_RES]], align 16
// LLVM: %[[TMP2:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: %[[SUB:.*]] = sub <4 x i32> zeroinitializer, %[[TMP2]]
// LLVM: store <4 x i32> %[[SUB]], ptr %[[MINUS_RES]], align 16
// LLVM: %[[TMP3:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// LLVM: %[[NOT:.*]] = xor <4 x i32> %[[TMP3]], splat (i32 -1)
// LLVM: store <4 x i32> %[[NOT]], ptr %[[NOT_RES]], align 16

// OGCG: %[[VEC:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[PLUS_RES:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[MINUS_RES:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[NOT_RES:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC]], align 16
// OGCG: %[[TMP1:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: store <4 x i32> %[[TMP1]], ptr %[[PLUS_RES]], align 16
// OGCG: %[[TMP2:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: %[[SUB:.*]] = sub <4 x i32> zeroinitializer, %[[TMP2]]
// OGCG: store <4 x i32> %[[SUB]], ptr %[[MINUS_RES]], align 16
// OGCG: %[[TMP3:.*]] = load <4 x i32>, ptr %[[VEC]], align 16
// OGCG: %[[NOT:.*]] = xor <4 x i32> %[[TMP3]], splat (i32 -1)
// OGCG: store <4 x i32> %[[NOT]], ptr %[[NOT_RES]], align 16

void foo9() {
  vi4 a = {1, 2, 3, 4};
  vi4 b = {5, 6, 7, 8};

  vi4 shl = a << b;
  vi4 shr = a >> b;
}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[VEC_B:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["b", init]
// CIR: %[[SHL_RES:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["shl", init]
// CIR: %[[SHR_RES:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["shr", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_A_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_A_VAL]], %[[VEC_A]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_5:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[CONST_6:.*]] = cir.const #cir.int<6> : !s32i
// CIR: %[[CONST_7:.*]] = cir.const #cir.int<7> : !s32i
// CIR: %[[CONST_8:.*]] = cir.const #cir.int<8> : !s32i
// CIR: %[[VEC_B_VAL:.*]] = cir.vec.create(%[[CONST_5]], %[[CONST_6]], %[[CONST_7]], %[[CONST_8]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_B_VAL]], %[[VEC_B]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[SHL:.*]] = cir.shift(left, %[[TMP_A]] : !cir.vector<4 x !s32i>, %[[TMP_B]] : !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>
// CIR: cir.store %[[SHL]], %[[SHL_RES]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[SHR:.*]] = cir.shift(right, %[[TMP_A]] : !cir.vector<4 x !s32i>, %[[TMP_B]] : !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>
// CIR: cir.store %[[SHR]], %[[SHR_RES]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_B:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[SHL_RES:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[SHR_RES:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_A]], align 16
// LLVM: store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %[[VEC_B]], align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[SHL:.*]] = shl <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[SHL]], ptr %[[SHL_RES]], align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[SHR:.*]] = ashr <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[SHR]], ptr %[[SHR_RES]], align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_B:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[SHL_RES:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[SHR_RES:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_A]], align 16
// OGCG: store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %[[VEC_B]], align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[SHL:.*]] = shl <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[SHL]], ptr %[[SHL_RES]], align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[SHR:.*]] = ashr <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[SHR]], ptr %[[SHR_RES]], align 16

void foo11() {
  vi4 a = {1, 2, 3, 4};
  vi4 b = {5, 6, 7, 8};

  vi4 c = a + b;
  vi4 d = a - b;
  vi4 e = a * b;
  vi4 f = a / b;
  vi4 g = a % b;
  vi4 h = a & b;
  vi4 i = a | b;
  vi4 j = a ^ b;
}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[VEC_B:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["b", init]
// CIR: %[[VEC_A_VAL:.*]] = cir.vec.create({{.*}}, {{.*}}, {{.*}}, {{.*}} : !s32i, !s32i, !s32i, !s32i) :
// CIR-SAME: !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_A_VAL]], %[[VEC_A]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[VEC_B_VAL:.*]] = cir.vec.create({{.*}}, {{.*}}, {{.*}}, {{.*}} : !s32i, !s32i, !s32i, !s32i) :
// CIR-SAME: !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_B_VAL]], %[[VEC_B]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[ADD:.*]] = cir.binop(add, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[ADD]], {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[SUB:.*]] = cir.binop(sub, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[SUB]], {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[MUL:.*]] = cir.binop(mul, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[MUL]], {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[DIV:.*]] = cir.binop(div, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[DIV]], {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[REM:.*]] = cir.binop(rem, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[REM]], {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[AND:.*]] = cir.binop(and, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[AND]], {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[OR:.*]] = cir.binop(or, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[OR]], {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load %[[VEC_A]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load %[[VEC_B]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[XOR:.*]] = cir.binop(xor, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[XOR]], {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_B:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_A]], align 16
// LLVM: store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %[[VEC_B]], align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[ADD:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[ADD]], ptr {{.*}}, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[SUB:.*]] = sub <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[SUB]], ptr {{.*}}, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[MUL:.*]] = mul <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[MUL]], ptr {{.*}}, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[DIV:.*]] = sdiv <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[DIV]], ptr {{.*}}, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[REM:.*]] = srem <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[REM]], ptr {{.*}}, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[AND:.*]] = and <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[AND]], ptr {{.*}}, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[OR:.*]] = or <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[OR]], ptr {{.*}}, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// LLVM: %[[XOR:.*]] = xor <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[XOR]], ptr {{.*}}, align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_B:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_A]], align 16
// OGCG: store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %[[VEC_B]], align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[ADD:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[ADD]], ptr {{.*}}, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[SUB:.*]] = sub <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[SUB]], ptr {{.*}}, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[MUL:.*]] = mul <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[MUL]], ptr {{.*}}, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[DIV:.*]] = sdiv <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[DIV]], ptr {{.*}}, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[REM:.*]] = srem <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[REM]], ptr {{.*}}, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[AND:.*]] = and <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[AND]], ptr {{.*}}, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[OR:.*]] = or <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[OR]], ptr {{.*}}, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[VEC_A]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[VEC_B]], align 16
// OGCG: %[[XOR:.*]] = xor <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[XOR]], ptr {{.*}}, align 16
