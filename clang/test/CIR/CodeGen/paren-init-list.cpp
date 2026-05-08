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

struct HasDtor {
  int val;
  ~HasDtor();
};

struct Outer {
  HasDtor h;
  int x;
};

void test_init_list_with_dtor() {
  Outer o = {HasDtor{1}, 2};
}

// CIR: cir.func {{.*}} @_Z24test_init_list_with_dtorv
// CIR:   %[[O:.*]] = cir.alloca !rec_Outer, !cir.ptr<!rec_Outer>, ["o", init]
// CIR:   %[[H:.*]] = cir.get_member %[[O]][0] {name = "h"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!rec_HasDtor>
// CIR:   %[[VAL:.*]] = cir.get_member %[[H]][0] {name = "val"} : !cir.ptr<!rec_HasDtor> -> !cir.ptr<!s32i>
// CIR:   %[[CONST:.*]] = cir.const #cir.int<1>
// CIR:   cir.store{{.*}} %[[CONST]], %[[VAL]]
// CIR:   %[[X:.*]] = cir.get_member %[[O]][1] {name = "x"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!s32i>
// CIR:   %[[CONST:.*]] = cir.const #cir.int<2>
// CIR:   cir.store{{.*}} %[[CONST]], %[[X]]
// CIR:   cir.cleanup.scope {
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     cir.call @_ZN5OuterD1Ev(%[[O]]) nothrow : (!cir.ptr<!rec_Outer> {llvm.align = 4 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define {{.*}} void @_Z24test_init_list_with_dtorv
// LLVM:   %[[O:.*]] = alloca %struct.Outer
// LLVM:   %[[O_ADDR:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %[[O]], i32 0, i32 0
// LLVM:   %[[H_ADDR:.*]] = getelementptr inbounds nuw %struct.HasDtor, ptr %[[O_ADDR]], i32 0, i32 0
// LLVM:   store i32 1, ptr %[[H_ADDR]]
// LLVM:   %[[X_ADDR:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %[[O]], i32 0, i32 1
// LLVM:   store i32 2, ptr %[[X_ADDR]]
// LLVM:   call void @_ZN5OuterD1Ev(ptr{{.*}} %[[O]])
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z24test_init_list_with_dtorv
// OGCG:   %[[O:.*]] = alloca %struct.Outer
// OGCG:   %[[O_ADDR:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %[[O]], i32 0, i32 0
// OGCG:   %[[H_ADDR:.*]] = getelementptr inbounds nuw %struct.HasDtor, ptr %[[O_ADDR]], i32 0, i32 0
// OGCG:   store i32 1, ptr %[[H_ADDR]]
// OGCG:   %[[X_ADDR:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %[[O]], i32 0, i32 1
// OGCG:   store i32 2, ptr %[[X_ADDR]]
// OGCG:   call void @_ZN5OuterD1Ev(ptr{{.*}} %[[O]])
// OGCG:   ret void
