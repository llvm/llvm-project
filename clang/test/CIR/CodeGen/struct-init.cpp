// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  int a, b, c;
};

void init() {
  S s1 = {1, 2, 3};
  S s2 = {4, 5};
}

// CIR: cir.func{{.*}} @_Z4initv()
// CIR:   %[[S1:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s1", init]
// CIR:   %[[S2:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s2", init]
// CIR:   %[[S1_A:.*]] = cir.get_member %[[S1]][0] {name = "a"}
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:   cir.store{{.*}} %[[ONE]], %[[S1_A]]
// CIR:   %[[S1_B:.*]] = cir.get_member %[[S1]][1] {name = "b"}
// CIR:   %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:   cir.store{{.*}} %[[TWO]], %[[S1_B]]
// CIR:   %[[S1_C:.*]] = cir.get_member %[[S1]][2] {name = "c"}
// CIR:   %[[THREE:.*]] = cir.const #cir.int<3>
// CIR:   cir.store{{.*}} %[[THREE]], %[[S1_C]]
// CIR:   %[[S2_A:.*]] = cir.get_member %[[S2]][0] {name = "a"}
// CIR:   %[[FOUR:.*]] = cir.const #cir.int<4>
// CIR:   cir.store{{.*}} %[[FOUR]], %[[S2_A]]
// CIR:   %[[S2_B:.*]] = cir.get_member %[[S2]][1] {name = "b"}
// CIR:   %[[FIVE:.*]] = cir.const #cir.int<5>
// CIR:   cir.store{{.*}} %[[FIVE]], %[[S2_B]]
// CIR:   %[[S2_C:.*]] = cir.get_member %[[S2]][2] {name = "c"}
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0>
// CIR:   cir.store{{.*}} %[[ZERO]], %[[S2_C]]
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z4initv()
// LLVM:   %[[S1:.*]] = alloca %struct.S
// LLVM:   %[[S2:.*]] = alloca %struct.S
// LLVM:   %[[S1_A:.*]] = getelementptr %struct.S, ptr %[[S1]], i32 0, i32 0
// LLVM:   store i32 1, ptr %[[S1_A]]
// LLVM:   %[[S1_B:.*]] = getelementptr %struct.S, ptr %[[S1]], i32 0, i32 1
// LLVM:   store i32 2, ptr %[[S1_B]]
// LLVM:   %[[S1_C:.*]] = getelementptr %struct.S, ptr %[[S1]], i32 0, i32 2
// LLVM:   store i32 3, ptr %[[S1_C]]
// LLVM:   %[[S2_A:.*]] = getelementptr %struct.S, ptr %[[S2]], i32 0, i32 0
// LLVM:   store i32 4, ptr %[[S2_A]]
// LLVM:   %[[S2_B:.*]] = getelementptr %struct.S, ptr %[[S2]], i32 0, i32 1
// LLVM:   store i32 5, ptr %[[S2_B]]
// LLVM:   %[[S2_C:.*]] = getelementptr %struct.S, ptr %[[S2]], i32 0, i32 2
// LLVM:   store i32 0, ptr %[[S2_C]]

// OGCG: @__const._Z4initv.s1 = private unnamed_addr constant %struct.S { i32 1, i32 2, i32 3 }
// OGCG: @__const._Z4initv.s2 = private unnamed_addr constant %struct.S { i32 4, i32 5, i32 0 }

// OGCG: define{{.*}} void @_Z4initv()
// OGCG:   %[[S1:.*]] = alloca %struct.S
// OGCG:   %[[S2:.*]] = alloca %struct.S
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[S1]], ptr{{.*}} @__const._Z4initv.s1, i64 12, i1 false)
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[S2]], ptr{{.*}} @__const._Z4initv.s2, i64 12, i1 false)

void init_var(int a, int b) {
  S s = {a, b};
}

// CIR: cir.func{{.*}} @_Z8init_varii(%[[A_ARG:.*]]: !s32i {{.*}}, %[[B_ARG:.*]]: !s32i {{.*}})
// CIR:   %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR:   %[[B_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR:   %[[S:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init]
// CIR:   cir.store{{.*}} %[[A_ARG]], %[[A_PTR]]
// CIR:   cir.store{{.*}} %[[B_ARG]], %[[B_PTR]]
// CIR:   %[[S_A:.*]] = cir.get_member %[[S]][0] {name = "a"}
// CIR:   %[[A:.*]] = cir.load{{.*}} %[[A_PTR]]
// CIR:   cir.store{{.*}} %[[A]], %[[S_A]]
// CIR:   %[[S_B:.*]] = cir.get_member %[[S]][1] {name = "b"}
// CIR:   %[[B:.*]] = cir.load{{.*}} %[[B_PTR]]
// CIR:   cir.store{{.*}} %[[B]], %[[S_B]]
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z8init_varii(i32 %[[A_ARG:.*]], i32 %[[B_ARG:.*]])
// LLVM:   %[[A_PTR:.*]] = alloca i32
// LLVM:   %[[B_PTR:.*]] = alloca i32
// LLVM:   %[[S:.*]] = alloca %struct.S
// LLVM:   store i32 %[[A_ARG]], ptr %[[A_PTR]]
// LLVM:   store i32 %[[B_ARG]], ptr %[[B_PTR]]
// LLVM:   %[[S_A:.*]] = getelementptr %struct.S, ptr %[[S]], i32 0, i32 0
// LLVM:   %[[A:.*]] = load i32, ptr %[[A_PTR]] 
// LLVM:   store i32 %[[A]], ptr %[[S_A]]
// LLVM:   %[[S_B:.*]] = getelementptr %struct.S, ptr %[[S]], i32 0, i32 1
// LLVM:   %[[B:.*]] = load i32, ptr %[[B_PTR]]
// LLVM:   store i32 %[[B]], ptr %[[S_B]]
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z8init_varii(i32 {{.*}} %[[A_ARG:.*]], i32 {{.*}} %[[B_ARG:.*]])
// OGCG:   %[[A_PTR:.*]] = alloca i32
// OGCG:   %[[B_PTR:.*]] = alloca i32
// OGCG:   %[[S:.*]] = alloca %struct.S
// OGCG:   store i32 %[[A_ARG]], ptr %[[A_PTR]]
// OGCG:   store i32 %[[B_ARG]], ptr %[[B_PTR]]
// OGCG:   %[[S_A:.*]] = getelementptr {{.*}} %struct.S, ptr %[[S]], i32 0, i32 0
// OGCG:   %[[A:.*]] = load i32, ptr %[[A_PTR]] 
// OGCG:   store i32 %[[A]], ptr %[[S_A]]
// OGCG:   %[[S_B:.*]] = getelementptr {{.*}} %struct.S, ptr %[[S]], i32 0, i32 1
// OGCG:   %[[B:.*]] = load i32, ptr %[[B_PTR]]
// OGCG:   store i32 %[[B]], ptr %[[S_B]]
// OGCG:   %[[S_C:.*]] = getelementptr {{.*}} %struct.S, ptr %[[S]], i32 0, i32 2
// OGCG:   store i32 0, ptr %[[S_C]]
// OGCG:   ret void

void init_expr(int a, int b, int c) {
  S s = {a + 1, b + 2, c + 3};
}

// CIR: cir.func{{.*}} @_Z9init_expriii(%[[A_ARG:.*]]: !s32i {{.*}}, %[[B_ARG:.*]]: !s32i {{.*}}, %[[C_ARG:.*]]: !s32i {{.*}})
// CIR:   %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR:   %[[B_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR:   %[[C_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c", init]
// CIR:   %[[S:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init]
// CIR:   cir.store{{.*}} %[[A_ARG]], %[[A_PTR]]
// CIR:   cir.store{{.*}} %[[B_ARG]], %[[B_PTR]]
// CIR:   cir.store{{.*}} %[[C_ARG]], %[[C_PTR]]
// CIR:   %[[S_A:.*]] = cir.get_member %[[S]][0] {name = "a"}
// CIR:   %[[A:.*]] = cir.load{{.*}} %[[A_PTR]]
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:   %[[A_PLUS_ONE:.*]] = cir.binop(add, %[[A]], %[[ONE]])
// CIR:   cir.store{{.*}} %[[A_PLUS_ONE]], %[[S_A]]
// CIR:   %[[S_B:.*]] = cir.get_member %[[S]][1] {name = "b"}
// CIR:   %[[B:.*]] = cir.load{{.*}} %[[B_PTR]]
// CIR:   %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:   %[[B_PLUS_TWO:.*]] = cir.binop(add, %[[B]], %[[TWO]]) nsw : !s32i
// CIR:   cir.store{{.*}} %[[B_PLUS_TWO]], %[[S_B]]
// CIR:   %[[S_C:.*]] = cir.get_member %[[S]][2] {name = "c"}
// CIR:   %[[C:.*]] = cir.load{{.*}} %[[C_PTR]]
// CIR:   %[[THREE:.*]] = cir.const #cir.int<3>
// CIR:   %[[C_PLUS_THREE:.*]] = cir.binop(add, %[[C]], %[[THREE]]) nsw : !s32i
// CIR:   cir.store{{.*}} %[[C_PLUS_THREE]], %[[S_C]]
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z9init_expriii(i32 %[[A_ARG:.*]], i32 %[[B_ARG:.*]], i32 %[[C_ARG:.*]])
// LLVM:   %[[A_PTR:.*]] = alloca i32
// LLVM:   %[[B_PTR:.*]] = alloca i32
// LLVM:   %[[C_PTR:.*]] = alloca i32
// LLVM:   %[[S:.*]] = alloca %struct.S
// LLVM:   store i32 %[[A_ARG]], ptr %[[A_PTR]]
// LLVM:   store i32 %[[B_ARG]], ptr %[[B_PTR]]
// LLVM:   store i32 %[[C_ARG]], ptr %[[C_PTR]]
// LLVM:   %[[S_A:.*]] = getelementptr %struct.S, ptr %[[S]], i32 0, i32 0
// LLVM:   %[[A:.*]] = load i32, ptr %[[A_PTR]] 
// LLVM:   %[[A_PLUS_ONE:.*]] = add nsw i32 %[[A]], 1
// LLVM:   store i32 %[[A_PLUS_ONE]], ptr %[[S_A]]
// LLVM:   %[[S_B:.*]] = getelementptr %struct.S, ptr %[[S]], i32 0, i32 1
// LLVM:   %[[B:.*]] = load i32, ptr %[[B_PTR]]
// LLVM:   %[[B_PLUS_TWO:.*]] = add nsw i32 %[[B]], 2
// LLVM:   store i32 %[[B_PLUS_TWO]], ptr %[[S_B]]
// LLVM:   %[[S_C:.*]] = getelementptr %struct.S, ptr %[[S]], i32 0, i32 2
// LLVM:   %[[C:.*]] = load i32, ptr %[[C_PTR]]
// LLVM:   %[[C_PLUS_THREE:.*]] = add nsw i32 %[[C]], 3
// LLVM:   store i32 %[[C_PLUS_THREE]], ptr %[[S_C]]
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z9init_expriii(i32 {{.*}} %[[A_ARG:.*]], i32 {{.*}} %[[B_ARG:.*]], i32 {{.*}} %[[C_ARG:.*]])
// OGCG:   %[[A_PTR:.*]] = alloca i32
// OGCG:   %[[B_PTR:.*]] = alloca i32
// OGCG:   %[[C_PTR:.*]] = alloca i32
// OGCG:   %[[S:.*]] = alloca %struct.S
// OGCG:   store i32 %[[A_ARG]], ptr %[[A_PTR]]
// OGCG:   store i32 %[[B_ARG]], ptr %[[B_PTR]]
// OGCG:   store i32 %[[C_ARG]], ptr %[[C_PTR]]
// OGCG:   %[[S_A:.*]] = getelementptr {{.*}} %struct.S, ptr %[[S]], i32 0, i32 0
// OGCG:   %[[A:.*]] = load i32, ptr %[[A_PTR]] 
// OGCG:   %[[A_PLUS_ONE:.*]] = add nsw i32 %[[A]], 1
// OGCG:   store i32 %[[A_PLUS_ONE]], ptr %[[S_A]]
// OGCG:   %[[S_B:.*]] = getelementptr {{.*}} %struct.S, ptr %[[S]], i32 0, i32 1
// OGCG:   %[[B:.*]] = load i32, ptr %[[B_PTR]]
// OGCG:   %[[B_PLUS_TWO:.*]] = add nsw i32 %[[B]], 2
// OGCG:   store i32 %[[B_PLUS_TWO]], ptr %[[S_B]]
// OGCG:   %[[S_C:.*]] = getelementptr {{.*}} %struct.S, ptr %[[S]], i32 0, i32 2
// OGCG:   %[[C:.*]] = load i32, ptr %[[C_PTR]]
// OGCG:   %[[C_PLUS_THREE:.*]] = add nsw i32 %[[C]], 3
// OGCG:   store i32 %[[C_PLUS_THREE]], ptr %[[S_C]]
// OGCG:   ret void
