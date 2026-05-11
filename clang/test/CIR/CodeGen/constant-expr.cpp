// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct StructWithConstEval {
  consteval int _Complex consteval_ret_complex() { return {1, 2}; }
  consteval int consteval_ret_int() { return 1; }
  consteval void consteval_ret_void() {}
};

void calling_consteval_methods() {
  StructWithConstEval a;
  int b = a.consteval_ret_int();
  int _Complex c = a.consteval_ret_complex();
  a.consteval_ret_void();
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_StructWithConstEval, !cir.ptr<!rec_StructWithConstEval>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store {{.*}} %[[CONST_1]], %[[B_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[CONST_COMPLEX:.*]] = cir.const #cir.const_complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>
// CIR: cir.store {{.*}} %[[CONST_COMPLEX]], %[[C_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.StructWithConstEval, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[C_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store i32 1, ptr %[[B_ADDR]], align 4
// LLVM: store { i32, i32 } { i32 1, i32 2 }, ptr %[[C_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca %struct.StructWithConstEval, align 1
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[C_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: store i32 1, ptr %[[B_ADDR]], align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG: store i32 1, ptr %[[C_REAL_PTR]], align 4
// OGCG: store i32 2, ptr %[[C_IMAG_PTR]], align 4

