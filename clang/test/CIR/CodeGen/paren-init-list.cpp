// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct CompleteS {
  int a;
  char b;
};

void cxx_paren_list_init_expr() { CompleteS a(1, 'a'); }

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a", init]
// CIR: %[[ELEM_0_PTR:.*]] = cir.get_member %[[A_ADDR]][0] {name = "a"} : !cir.ptr<!rec_CompleteS> -> !cir.ptr<!s32i>
// CIR: %[[ELEM_0_VAL:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[ELEM_0_VAL]], %[[ELEM_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[ELEM_1_PTR:.*]] = cir.get_member %[[A_ADDR]][1] {name = "b"} : !cir.ptr<!rec_CompleteS> -> !cir.ptr<!s8i>
// CIR: %[[ELEM_1_VAL:.*]] = cir.const #cir.int<97> : !s8i
// CIR: cir.store{{.*}} %[[ELEM_1_VAL]], %[[ELEM_1_PTR]] : !s8i, !cir.ptr<!s8i>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM: %[[ELEM_0_PTR:.*]] = getelementptr %struct.CompleteS, ptr %[[A_ADDR]], i32 0, i32 0
// LLVM: store i32 1, ptr %[[ELEM_0_PTR]], align 4
// LLVM: %[[ELEM_1_PTR:.*]] = getelementptr %struct.CompleteS, ptr %[[A_ADDR]], i32 0, i32 1
// LLVM: store i8 97, ptr %[[ELEM_1_PTR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[A_ADDR]], ptr align 4 @__const._Z24cxx_paren_list_init_exprv.a, i64 8, i1 false)
