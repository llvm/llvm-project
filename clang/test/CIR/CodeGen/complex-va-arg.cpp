// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// TODO(CIR): Without calling convention lowering, CIR lowers this va_arg to an
// LLVM va_arg of { float, float }, which the verifier rejects. Add the
// CIR-to-LLVM checks back once it is lowered the way OGCG does below.

void foo33(__builtin_va_list a) {
  float _Complex b = __builtin_va_arg(a, float _Complex);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.ptr<!rec___va_list_tag>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.complex<!cir.float>>
// CIR: cir.store %[[ARG_0:.*]], %[[A_ADDR]] : !cir.ptr<!rec___va_list_tag>, !cir.ptr<!cir.ptr<!rec___va_list_tag>>
// CIR: %[[VA_TAG:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.ptr<!rec___va_list_tag>>, !cir.ptr<!rec___va_list_tag>
// CIR: %[[COMPLEX:.*]] = cir.va_arg %[[VA_TAG]] : (!cir.ptr<!rec___va_list_tag>) -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// OGCG: %[[A_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: store ptr %[[ARG_0:.*]], ptr %[[A_ADDR]], align 8
// OGCG: %[[TMP_A:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// OGCG: %[[GP_OFFSET_PTR:.*]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[TMP_A]], i32 0, i32 1
// OGCG: %[[GP_OFFSET:.*]] = load i32, ptr %[[GP_OFFSET_PTR]], align 4
// OGCG: %[[COND:.*]] = icmp ule i32 %[[GP_OFFSET]], 160
// OGCG: br i1 %[[COND]], label %[[VA_ARG_IN_REG:.*]], label %[[VA_ARG_IN_MEM:.*]]
//
// OGCG: [[VA_ARG_IN_REG]]:
// OGCG:  %[[REG_SAVE_PTR:.*]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[TMP_A]], i32 0, i32 3
// OGCG:  %[[REG_SAVE:.*]] = load ptr, ptr %[[REG_SAVE_PTR]], align 8
// OGCG:  %[[VA_ADDR:..*]] = getelementptr i8, ptr %[[REG_SAVE]], i32 %[[GP_OFFSET]]
// OGCG:  %[[GP_OFFSET_NEXT:.*]] = add i32 %[[GP_OFFSET]], 16
// OGCG:  store i32 %[[GP_OFFSET_NEXT]], ptr %[[GP_OFFSET_PTR]], align 4
// OGCG:  br label %[[VA_ARG_END:.*]]
//
// OGCG: [[VA_ARG_IN_MEM]]:
// OGCG:  %[[OVERFLOW_PTR:.*]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[TMP_A]], i32 0, i32 2
// OGCG:  %[[OVERFLOW:.*]] = load ptr, ptr %[[OVERFLOW_PTR]], align 8
// OGCG:  %[[OVERFLOW_NEXT:.*]] = getelementptr i8, ptr %[[OVERFLOW]], i32 8
// OGCG:  store ptr %[[OVERFLOW_NEXT]], ptr %[[OVERFLOW_PTR]], align 8
// OGCG:  br label %[[VA_ARG_END]]
//
// OGCG: [[VA_ARG_END]]:
// OGCG:  %[[RESULT:.*]] = phi ptr [ %[[VA_ADDR]], %[[VA_ARG_IN_REG]] ], [ %[[OVERFLOW]], %[[VA_ARG_IN_MEM]] ]
// OGCG:  %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 0
// OGCG:  %[[RESULT_REAL:.*]] = load float, ptr %[[RESULT_REAL_PTR]], align 4
// OGCG:  %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 1
// OGCG:  %[[RESULT_IMAG:.*]] = load float, ptr %[[RESULT_IMAG_PTR]], align 4
// OGCG:  %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG:  %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG:  store float %[[RESULT_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG:  store float %[[RESULT_IMAG]], ptr %[[B_IMAG_PTR]], align 4
