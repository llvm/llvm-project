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

// CIR-DAG: cir.global "private" constant cir_private @[[PAREN_A:.*]] = #cir.const_record<{#cir.int<1> : !s32i, #cir.int<97> : !s8i}> : !rec_CompleteS
// LLVM-DAG: @[[PAREN_A:.*]] = private constant %struct.CompleteS { i32 1, i8 97 }

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a", init]
// CIR: %[[CONST:.*]] = cir.get_global @[[PAREN_A]] : !cir.ptr<!rec_CompleteS>
// CIR: cir.copy %[[CONST]] to %[[A_ADDR]]

// LLVM: %[[A_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM: call void @llvm.memcpy{{.*}}(ptr %[[A_ADDR]], ptr @[[PAREN_A]]

// OGCG: %[[A_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[A_ADDR]], ptr align 4 @__const._Z24cxx_paren_list_init_exprv.a, i64 8, i1 false)
