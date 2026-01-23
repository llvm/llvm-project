// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

template <typename T> void summable(T a) {
  if (requires { a + a; }) {
    T b = a + a;
  }
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: cir.store %[[ARG_A:.*]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.scope {
// CIR:   %[[CONST_TRUE:.*]] = cir.const #true
// CIR:   cir.if %[[CONST_TRUE]] {
// CIR:     %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR:     %[[TMP_A_1:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[TMP_A_2:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.binop(add, %[[TMP_A_1]], %[[TMP_A_2]]) nsw : !s32i
// CIR:     cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   }
// CIR: }

// LLVM:   %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 %[[ARG_A:.*]], ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[IF_COND:.*]]
// LLVM: [[IF_COND]]:
// LLVM:   br i1 true, label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// LLVM: [[IF_THEN]]:
// LLVM:   %[[TMP_A_1:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[TMP_A_2:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[TMP_A_1]], %[[TMP_A_2]]
// LLVM:   store i32 %[[RESULT]], ptr %[[B_ADDR]], align 4
// LLVM:   br label %[[IF_END]]
// LLVM: [[IF_END]]:
// LLVM:   br label %[[RET:.*]]

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 %[[ARG_A:.*]], ptr %[[A_ADDR]], align 4
// OGCG: %[[TMP_A_1:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: %[[TMP_A_2:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: %[[RESULT:.*]] = add nsw i32 %[[TMP_A_1]], %[[TMP_A_2]]
// OGCG: store i32 %[[RESULT]], ptr %[[B_ADDR]], align 4

void call_function_with_requires_expr() { summable(1); }

