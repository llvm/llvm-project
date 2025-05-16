// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef int vi4 __attribute__((vector_size(16)));
typedef double vd2 __attribute__((vector_size(16)));
typedef long long vll2 __attribute__((vector_size(16)));

vi4 vec_a;
// CIR: cir.global external @[[VEC_A:.*]] = #cir.zero : !cir.vector<4 x !s32i>

// LLVM: @[[VEC_A:.*]] = dso_local global <4 x i32> zeroinitializer

// OGCG: @[[VEC_A:.*]] = global <4 x i32> zeroinitializer

vd2 b;
// CIR: cir.global external @[[VEC_B:.*]] = #cir.zero : !cir.vector<2 x !cir.double>

// LLVM: @[[VEC_B:.*]] = dso_local global <2 x double> zeroinitialize

// OGCG: @[[VEC_B:.*]] = global <2 x double> zeroinitializer

vll2 c;
// CIR: cir.global external @[[VEC_C:.*]] = #cir.zero : !cir.vector<2 x !s64i>

// LLVM: @[[VEC_C:.*]] = dso_local global <2 x i64> zeroinitialize

// OGCG: @[[VEC_C:.*]] = global <2 x i64> zeroinitializer

vi4 d = { 1, 2, 3, 4 };

// CIR: cir.global external @[[VEC_D:.*]] = #cir.const_vector<[#cir.int<1> : !s32i, #cir.int<2> :
// CIR-SAME: !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.vector<4 x !s32i>

// LLVM: @[[VEC_D:.*]] = dso_local global <4 x i32> <i32 1, i32 2, i32 3, i32 4>

// OGCG: @[[VEC_D:.*]] = global <4 x i32> <i32 1, i32 2, i32 3, i32 4>

int x = 5;

void foo() {
  vi4 a;
  vd2 b;
  vll2 c;

  vi4 d = { 1, 2, 3, 4 };

  vi4 e = { x, 5, 6, x + 1 };

  vi4 f = { 5 };

  vi4 g = {};
}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[VEC_B:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["b"]
// CIR: %[[VEC_C:.*]] = cir.alloca !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>, ["c"]
// CIR: %[[VEC_D:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["d", init]
// CIR: %[[VEC_E:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["e", init]
// CIR: %[[VEC_F:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["f", init]
// CIR: %[[VEC_G:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["g", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !s32i
// CIR: %[[VEC_D_VAL:.*]] = cir.vec.create(%[[CONST_1]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_D_VAL]], %[[VEC_D]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[GLOBAL_X:.*]] = cir.get_global @x : !cir.ptr<!s32i>
// CIR: %[[X_VAL:.*]] = cir.load %[[GLOBAL_X]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[CONST_5:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[CONST_6:.*]] = cir.const #cir.int<6> : !s32i
// CIR: %[[GLOBAL_X:.*]] = cir.get_global @x : !cir.ptr<!s32i>
// CIR: %[[X:.*]] = cir.load %[[GLOBAL_X]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[X_PLUS_1:.*]] = cir.binop(add, %[[X]], %[[CONST_1]]) nsw : !s32i
// CIR: %[[VEC_E_VAL:.*]] = cir.vec.create(%[[X_VAL]], %[[CONST_5]], %[[CONST_6]], %[[X_PLUS_1]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_E_VAL]], %[[VEC_E]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_5:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[VEC_F_VAL:.*]] = cir.vec.create(%[[CONST_5]], %[[CONST_0]], %[[CONST_0]], %[[CONST_0]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_F_VAL]], %[[VEC_F]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[VEC_G_VAL:.*]] = cir.vec.create(%[[CONST_0]], %[[CONST_0]], %[[CONST_0]], %[[CONST_0]] :
// CIR-SAME: !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_G_VAL]], %[[VEC_G]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_B:.*]] = alloca <2 x double>, i64 1, align 16
// LLVM: %[[VEC_C:.*]] = alloca <2 x i64>, i64 1, align 16
// LLVM: %[[VEC_D:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_E:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_F:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_G:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_D]], align 16
// LLVM: store <4 x i32> {{.*}}, ptr %[[VEC_E]], align 16
// LLVM: store <4 x i32> <i32 5, i32 0, i32 0, i32 0>, ptr %[[VEC_F]], align 16
// LLVM: store <4 x i32> zeroinitializer, ptr %[[VEC_G]], align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_B:.*]] = alloca <2 x double>, align 16
// OGCG: %[[VEC_C:.*]] = alloca <2 x i64>, align 16
// OGCG: %[[VEC_D:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_E:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_F:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_G:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_D]], align 16
// OGCG: store <4 x i32> {{.*}}, ptr %[[VEC_E]], align 16
// OGCG: store <4 x i32> <i32 5, i32 0, i32 0, i32 0>, ptr %[[VEC_F]], align 16
// OGCG: store <4 x i32> zeroinitializer, ptr %[[VEC_G]], align 16

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
