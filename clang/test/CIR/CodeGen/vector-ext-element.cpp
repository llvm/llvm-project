// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef int vi2 __attribute__((ext_vector_type(2)));
typedef int vi4 __attribute__((ext_vector_type(4)));

void element_expr_from_gl() {
  vi4 a;
  int x = a.x;
  int y = a.y;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[X_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR: %[[Y_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s64i
// CIR: %[[ELEM_0:.*]] = cir.vec.extract %[[TMP_A]][%[[CONST_0]] : !s64i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ELEM_0]], %[[X_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELEM_1:.*]] = cir.vec.extract %[[TMP_A]][%[[CONST_1]] : !s64i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ELEM_1]], %[[Y_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[X_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[Y_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[ELEM_0:.*]] = extractelement <4 x i32> %4, i64 0
// LLVM: store i32 %[[ELEM_0]], ptr %[[X_ADDR]], align 4
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[ELEM_1:.*]] = extractelement <4 x i32> %6, i64 1
// LLVM: store i32 %[[ELEM_1]], ptr %[[Y_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[X_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[Y_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[ELEM_0:.*]] = extractelement <4 x i32> %[[TMP_A]], i64 0
// OGCG: store i32 %[[ELEM_0]], ptr %[[X_ADDR]], align 4
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[ELEM_1:.*]] = extractelement <4 x i32> %[[TMP_A]], i64 1
// OGCG: store i32 %[[ELEM_1]], ptr %[[Y_ADDR]], align 4

void element_expr_from_gl_with_vec_result() {
  vi4 a;
  vi2 b = a.xy;
  vi4 c = a.wzyx;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["c", init]
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[B_VALUE:.*]] = cir.vec.shuffle(%[[TMP_A]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !s32i>
// CIR: cir.store {{.*}} %[[B_VALUE]], %[[B_ADDR]] : !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[C_VALUE:.*]] = cir.vec.shuffle(%[[TMP_A]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[C_VALUE]], %[[C_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[B_ADDR:.*]] = alloca <2 x i32>, i64 1, align 8
// LLVM: %[[C_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[B_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <2 x i32> <i32 0, i32 1>
// LLVM: store <2 x i32> %[[B_VALUE]], ptr %[[B_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[C_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// LLVM: store <4 x i32> %[[C_VALUE]], ptr %[[C_ADDR]], align 16

// OGCG: %[[A_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[B_ADDR:.*]] = alloca <2 x i32>, align 8
// OGCG: %[[C_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[B_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <2 x i32> <i32 0, i32 1>
// OGCG: store <2 x i32> %[[B_VALUE]], ptr %[[B_ADDR]], align 8
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[C_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// OGCG: store <4 x i32> %[[C_VALUE]], ptr %[[C_ADDR]], align 16

void element_expr_from_pointer() {
  vi4 *a;
  int X = a->x;
  int Y = a->y;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!cir.vector<4 x !s32i>>, !cir.ptr<!cir.ptr<!cir.vector<4 x !s32i>>>, ["a"]
// CIR: %[[X_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["X", init]
// CIR: %[[Y_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["Y", init]
// CIR: %[[TMP_A_PTR:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.ptr<!cir.vector<4 x !s32i>>>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[TMP_A_PTR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s64i
// CIR: %[[ELEM_0:.*]] = cir.vec.extract %[[TMP_A]][%[[CONST_0]] : !s64i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ELEM_0]], %[[X_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_A_PTR:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.ptr<!cir.vector<4 x !s32i>>>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[TMP_A_PTR:.*]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELEM_1:.*]] = cir.vec.extract %[[TMP_A]][%[[CONST_1]] : !s64i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ELEM_1]], %[[Y_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[X_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[Y_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[TMP_A_PTR:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[TMP_A_PTR]], align 16
// LLVM: %[[ELEM_0:.*]] = extractelement <4 x i32> %[[TMP_A]], i64 0
// LLVM: store i32 %[[ELEM_0]], ptr %[[X_ADDR]], align 4
// LLVM: %[[TMP_A_PTR:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[TMP_A_PTR]], align 16
// LLVM: %[[ELEM_1:.*]] = extractelement <4 x i32> %[[TMP_A]], i64 1
// LLVM: store i32 %[[ELEM_1]], ptr %[[Y_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[X_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[Y_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[TMP_A_PTR:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[TMP_A_PTR]], align 16
// OGCG: %[[ELEM_0:.*]] = extractelement <4 x i32> %[[TMP_A]], i64 0
// OGCG: store i32 %[[ELEM_0]], ptr %[[X_ADDR]], align 4
// OGCG: %[[TMP_A_PTR:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[TMP_A_PTR]], align 16
// OGCG: %[[ELEM_1:.*]] = extractelement <4 x i32> %[[TMP_A]], i64 1
// OGCG: store i32 %[[ELEM_1]], ptr %[[Y_ADDR]], align 4

void element_expr_from_pointer_with_vec_result() {
  vi4 *a;
  vi2 b = a->xy;
  vi4 c = a->wzyx;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!cir.vector<4 x !s32i>>, !cir.ptr<!cir.ptr<!cir.vector<4 x !s32i>>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["c", init]
// CIR: %[[TMP_A_PTR:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.ptr<!cir.vector<4 x !s32i>>>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[TMP_A_PTR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[B_VALUE:.*]] = cir.vec.shuffle(%[[TMP_A]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !s32i>
// CIR: cir.store {{.*}} %[[B_VALUE]], %[[B_ADDR]] : !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>
// CIR: %[[TMP_A_PTR:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.ptr<!cir.vector<4 x !s32i>>>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[TMP_A_PTR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[C_VALUE:.*]] = cir.vec.shuffle(%[[TMP_A]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[C_VALUE]], %[[C_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca <2 x i32>, i64 1, align 8
// LLVM: %[[C_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[TMP_A_PTR:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[TMP_A_PTR]], align 16
// LLVM: %[[B_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <2 x i32> <i32 0, i32 1>
// LLVM: store <2 x i32> %[[B_VALUE]], ptr %[[B_ADDR]], align 8
// LLVM: %[[TMP_A_PTR:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[TMP_A_PTR]], align 16
// LLVM: %[[C_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// LLVM: store <4 x i32> %[[C_VALUE]], ptr %[[C_ADDR]], align 16

// OGCG: %[[A_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[B_ADDR:.*]] = alloca <2 x i32>, align 8
// OGCG: %[[C_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[TMP_A_PTR:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[TMP_A_PTR]], align 16
// OGCG: %[[B_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <2 x i32> <i32 0, i32 1>
// OGCG: store <2 x i32> %[[B_VALUE]], ptr %[[B_ADDR]], align 8
// OGCG: %[[TMP_A_PTR:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[TMP_A_PTR]], align 16
// OGCG: %[[C_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// OGCG: store <4 x i32> %[[C_VALUE]], ptr %[[C_ADDR]], align 16

void element_expr_from_rvalue() {
  vi4 a;
  vi4 b;
  int x = (a + b).x;
  int y = (a + b).y;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["b"]
// CIR: %[[X_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR: %[[TMP_1_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["tmp"]
// CIR: %[[Y_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
// CIR: %[[TMP_2_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["tmp"]
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[ADD_A_B:.*]] = cir.binop(add, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ADD_A_B]], %[[TMP_1_ADDR:.*]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_1:.*]] = cir.load {{.*}} %[[TMP_1_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s64i
// CIR: %[[ELEM_0:.*]] = cir.vec.extract %[[TMP_1]][%[[CONST_0]] : !s64i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ELEM_0]], %[[X_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[ADD_A_B:.*]] = cir.binop(add, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ADD_A_B]], %[[TMP_2_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_2:.*]] = cir.load {{.*}} %[[TMP_2_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELEM_1:.*]] = cir.vec.extract %[[TMP_2]][%[[CONST_1]] : !s64i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ELEM_1]], %[[Y_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[B_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[X_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[TMP_1_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[Y_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[TMP_2_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[B_ADDR]], align 16
// LLVM: %[[ADD_A_B:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[ADD_A_B]], ptr %[[TMP_1_ADDR]], align 16
// LLVM: %[[TMP_1:.*]] = load <4 x i32>, ptr %[[TMP_1_ADDR]], align 16
// LLVM: %[[ELEM_0:.*]] = extractelement <4 x i32> %[[TMP_1]], i64 0
// LLVM: store i32 %[[ELEM_0]], ptr %[[X_ADDR]], align 4
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[B_ADDR]], align 16
// LLVM: %[[ADD_A_B:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[ADD_A_B]], ptr %[[TMP_2_ADDR]], align 16
// LLVM: %[[TMP_2:.*]] = load <4 x i32>, ptr %[[TMP_2_ADDR]], align 16
// LLVM: %[[ELEM_1:.*]] = extractelement <4 x i32> %[[TMP_2]], i64 1
// LLVM: store i32 %[[ELEM_1]], ptr %[[Y_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[B_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[X_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[TMP_1_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[Y_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[TMP_2_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[B_ADDR]], align 16
// OGCG: %[[ADD_A_B:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[ADD_A_B]], ptr %[[TMP_1_ADDR]], align 16
// OGCG: %[[TMP_1:.*]] = load <4 x i32>, ptr %[[TMP_1_ADDR]], align 16
// OGCG: %[[ELEM_0:.*]] = extractelement <4 x i32> %[[TMP_1]], i64 0
// OGCG: store i32 %[[ELEM_0]], ptr %[[X_ADDR]], align 4
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[B_ADDR]], align 16
// OGCG: %[[ADD_A_B:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[ADD_A_B]], ptr %[[TMP_2_ADDR]], align 16
// OGCG: %[[TMP_2:.*]] = load <4 x i32>, ptr %[[TMP_2_ADDR]], align 16
// OGCG: %[[ELEM_1:.*]] = extractelement <4 x i32> %[[TMP_2]], i64 1
// OGCG: store i32 %[[ELEM_1]], ptr %[[Y_ADDR]], align 4

void element_expr_from_rvalue_with_vec_result() {
  vi4 a;
  vi4 b;
  vi2 c = (a + b).xy;
  vi4 d = (a + b).wzyx;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["b"]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>, ["c", init]
// CIR: %[[TMP_1_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["tmp"]
// CIR: %[[D_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["d", init]
// CIR: %[[TMP_2_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["tmp"]
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[ADD_A_B:.*]] = cir.binop(add, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ADD_A_B]], %[[TMP_1_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_1:.*]] = cir.load {{.*}} %[[TMP_1_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[C_VALUE:.*]] = cir.vec.shuffle(%[[TMP_1]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !s32i>
// CIR: cir.store {{.*}} %[[C_VALUE]], %[[C_ADDR]] : !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[ADD_A_B:.*]] = cir.binop(add, %[[TMP_A]], %[[TMP_B]]) : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[ADD_A_B]], %[[TMP_2_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_2:.*]] = cir.load {{.*}} %[[TMP_2_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[D_VALUE:.*]] = cir.vec.shuffle(%[[TMP_2]], %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[D_VALUE]], %[[D_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[B_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[C_ADDR:.*]] = alloca <2 x i32>, i64 1, align 8
// LLVM: %[[TMP_1_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[D_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[TMP_2_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[B_ADDR]], align 16
// LLVM: %[[ADD_A_B:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[ADD_A_B]], ptr %[[TMP_1_ADDR]], align 16
// LLVM: %[[TMP_1:.*]] = load <4 x i32>, ptr %[[TMP_1_ADDR]], align 16
// LLVM: %[[C_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_1]], <4 x i32> poison, <2 x i32> <i32 0, i32 1>
// LLVM: store <2 x i32> %[[C_VALUE]], ptr %[[C_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[B_ADDR]], align 16
// LLVM: %[[ADD_A_B:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// LLVM: store <4 x i32> %[[ADD_A_B]], ptr %[[TMP_2_ADDR]], align 16
// LLVM: %[[TMP_2:.*]] = load <4 x i32>, ptr %[[TMP_2_ADDR]], align 16
// LLVM: %[[D_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_2]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// LLVM: store <4 x i32> %[[D_VALUE]], ptr %[[D_ADDR]], align 16

// OGCG: %[[A_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[B_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[C_ADDR:.*]] = alloca <2 x i32>, align 8
// OGCG: %[[TMP_1_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[D_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[TMP_2_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[B_ADDR]], align 16
// OGCG: %[[ADD_A_B:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[ADD_A_B]], ptr %[[TMP_1_ADDR]], align 16
// OGCG: %[[TMP_1:.*]] = load <4 x i32>, ptr %[[TMP_1_ADDR]], align 16
// OGCG: %[[C_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_1]], <4 x i32> poison, <2 x i32> <i32 0, i32 1>
// OGCG: store <2 x i32> %[[C_VALUE]], ptr %[[C_ADDR]], align 8
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[TMP_B:.*]] = load <4 x i32>, ptr %[[B_ADDR]], align 16
// OGCG: %[[ADD_A_B:.*]] = add <4 x i32> %[[TMP_A]], %[[TMP_B]]
// OGCG: store <4 x i32> %[[ADD_A_B]], ptr %[[TMP_2_ADDR]], align 16
// OGCG: %[[TMP_2:.*]] = load <4 x i32>, ptr %[[TMP_2_ADDR]], align 16
// OGCG: %[[D_VALUE:.*]] = shufflevector <4 x i32> %[[TMP_2]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// OGCG: store <4 x i32> %[[D_VALUE]], ptr %[[D_ADDR]], align 16

void array_subscript_expr_with_element_expr_base() {
  vi4 a;
  a.xyz[1] = 2;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[A_PTR:.*]] = cir.cast bitcast %0 : !cir.ptr<!cir.vector<4 x !s32i>> -> !cir.ptr<!s32i>
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s64i
// CIR: %[[VEC_MEMBER_EXPR:.*]] = cir.ptr_stride %[[A_PTR]], %[[CONST_0]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[VEC_ELEM_PTR:.*]] = cir.ptr_stride %[[VEC_MEMBER_EXPR]], %[[CONST_1]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR: cir.store {{.*}} %[[CONST_2]], %[[VEC_ELEM_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_MEMBER_EXPR:.*]] = getelementptr i32, ptr %[[A_ADDR]], i64 0
// LLVM: %[[VEC_ELEM_PTR:.*]] = getelementptr i32, ptr %[[VEC_MEMBER_EXPR]], i64 1
// LLVM: store i32 2, ptr %[[VEC_ELEM_PTR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_MEMBER_EXPR:.*]] = getelementptr inbounds i32, ptr %[[A_ADDR]], i64 0
// OGCG: %[[VEC_ELEM_PTR:.*]] = getelementptr inbounds i32, ptr %[[VEC_MEMBER_EXPR]], i64 1
// OGCG: store i32 2, ptr %[[VEC_ELEM_PTR]], align 4
