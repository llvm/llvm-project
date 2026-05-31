// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void complex_to_atomic_complex() {
  _Complex int a;
  _Atomic _Complex int b = a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 8
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM: store { i32, i32 } %[[TMP_A]], ptr %[[B_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[A_REAL]], ptr %[[B_REAL_PTR]], align 8
// OGCG: store i32 %[[A_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void atomic_complex_to_complex() {
  _Atomic _Complex int a;
  _Complex int b = a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[ATOMIC_TMP_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["atomic-temp"]
// CIR: %[[A_U64I:.*]] = cir.cast bitcast %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!u64i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} atomic(seq_cst) %[[A_U64I]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[ATOMIC_TMP_U64I:.*]] = cir.cast bitcast %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!u64i>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[ATOMIC_TMP_U64I]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_ATOMIC:.*]] = cir.load {{.*}} %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store {{.*}} %[[TMP_ATOMIC]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[ATOMIC_TMP_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 8
// LLVM: %[[TMP_A:.*]] = load atomic i64, ptr %[[A_ADDR]] seq_cst, align 8
// LLVM: store i64 %[[TMP_A]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: %[[TMP_ATOMIC:.*]] = load { i32, i32 }, ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: store { i32, i32 } %[[TMP_ATOMIC]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 8
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[ATOMIC_TMP_ADDR:.*]] = alloca { i32, i32 }, align 8
// OGCG: %[[TMP_A:.*]] = load atomic i64, ptr %[[A_ADDR]] seq_cst, align 8
// OGCG: store i64 %[[TMP_A]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// OGCG: %[[ATOMIC_TMP_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 0
// OGCG: %[[ATOMIC_TMP_REAL:.*]] = load i32, ptr %[[ATOMIC_TMP_REAL_PTR]], align 8
// OGCG: %[[ATOMIC_TMP_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 1
// OGCG: %[[ATOMIC_TMP_IMAG:.*]] = load i32, ptr %[[ATOMIC_TMP_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[ATOMIC_TMP_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store i32 %[[ATOMIC_TMP_IMAG]], ptr %[[B_IMAG_PTR]], align 4
