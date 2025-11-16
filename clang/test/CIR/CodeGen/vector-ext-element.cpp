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
