// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct BitfieldStruct {
  unsigned int a:4;
  unsigned int b:14;
  unsigned int c:14;
};

BitfieldStruct overlapping_init = { 3, 2, 1 };

// This is unintuitive. The bitfields are initialized using a struct of constants
// that maps to the bitfields but splits the value into bytes.

// CIR: cir.global external @overlapping_init = #cir.const_record<{#cir.int<35> : !u8i, #cir.int<0> : !u8i, #cir.int<4> : !u8i, #cir.int<0> : !u8i}> : !rec_anon_struct
// LLVM: @overlapping_init = global { i8, i8, i8, i8 } { i8 35, i8 0, i8 4, i8 0 }
// OGCG: @overlapping_init = global { i8, i8, i8, i8 } { i8 35, i8 0, i8 4, i8 0 }

struct S {
  int a, b, c;
};

S partial_init = { 1 };

// CIR: cir.global external @partial_init = #cir.const_record<{#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i}> : !rec_S
// LLVM: @partial_init = global %struct.S { i32 1, i32 0, i32 0 }
// OGCG: @partial_init = global %struct.S { i32 1, i32 0, i32 0 }

struct StructWithDefaultInit {
  int a = 2;
};

StructWithDefaultInit swdi = {};

// CIR: cir.global external @swdi = #cir.const_record<{#cir.int<2> : !s32i}> : !rec_StructWithDefaultInit
// LLVM: @swdi = global %struct.StructWithDefaultInit { i32 2 }, align 4
// OGCG: @swdi = global %struct.StructWithDefaultInit { i32 2 }, align 4

struct StructWithFieldInitFromConst {
  int a : 10;
  int b = a;
};

StructWithFieldInitFromConst swfifc = {};

// CIR: cir.global external @swfifc = #cir.zero : !rec_anon_struct
// LLVM: @swfifc = global { i8, i8, i32 } zeroinitializer, align 4
// OGCG: @swfifc = global { i8, i8, i32 } zeroinitializer, align 4

StructWithFieldInitFromConst swfifc2 = { 2 };

// CIR: cir.global external @swfifc2 = #cir.const_record<{#cir.int<2> : !u8i, #cir.int<0> : !u8i, #cir.int<2> : !s32i}> : !rec_anon_struct
// LLVM: @swfifc2 = global { i8, i8, i32 } { i8 2, i8 0, i32 2 }, align 4
// OGCG: @swfifc2 = global { i8, i8, i32 } { i8 2, i8 0, i32 2 }, align 4

void init() {
  S s1 = {1, 2, 3};
  S s2 = {4, 5};
}

// CIR: cir.func{{.*}} @_Z4initv()
// CIR:   %[[S1:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s1", init]
// CIR:   %[[S2:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s2", init]
// CIR:   %[[CONST_1:.*]] = cir.const #cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i}> : !rec_S
// CIR:   cir.store{{.*}} %[[CONST_1]], %[[S1]]
// CIR:   %[[CONST_2:.*]] = cir.const #cir.const_record<{#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<0> : !s32i}> : !rec_S
// CIR:   cir.store{{.*}} %[[CONST_2]], %[[S2]]

// LLVM: define{{.*}} void @_Z4initv()
// LLVM:   %[[S1:.*]] = alloca %struct.S
// LLVM:   %[[S2:.*]] = alloca %struct.S
// LLVM:   store %struct.S { i32 1, i32 2, i32 3 }, ptr %[[S1]], align 4
// LLVM:   store %struct.S { i32 4, i32 5, i32 0 }, ptr %[[S2]], align 4

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

void cxx_default_init_with_struct_field() {
  struct Parent {
    int getA();
    int a = getA();
  };
  Parent p = Parent{};
}

// CIR: %[[P_ADDR:.*]] = cir.alloca !rec_Parent, !cir.ptr<!rec_Parent>, ["p", init]
// CIR: %[[P_ELEM_0_PTR:.*]] = cir.get_member %[[P_ADDR]][0] {name = "a"} : !cir.ptr<!rec_Parent> -> !cir.ptr<!s32i>
// CIR: %[[METHOD_CALL:.*]] = cir.call @_ZZ34cxx_default_init_with_struct_fieldvEN6Parent4getAEv(%[[P_ADDR]]) : (!cir.ptr<!rec_Parent>) -> !s32i
// CIR: cir.store{{.*}} %[[METHOD_CALL]], %[[P_ELEM_0_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[P_ADDR:.*]] = alloca %struct.Parent, i64 1, align 4
// LLVM: %[[P_ELEM_0_PTR:.*]] = getelementptr %struct.Parent, ptr %[[P_ADDR]], i32 0, i32 0
// LLVM: %[[METHOD_CALL:.*]] = call i32 @_ZZ34cxx_default_init_with_struct_fieldvEN6Parent4getAEv(ptr %[[P_ADDR]])
// LLVM: store i32 %[[METHOD_CALL]], ptr %[[P_ELEM_0_PTR]], align 4

// OGCG: %[[P_ADDR:.*]] = alloca %struct.Parent, align 4
// OGCG: %[[P_ELEM_0_PTR:.*]] = getelementptr inbounds nuw %struct.Parent, ptr %[[P_ADDR]], i32 0, i32 0
// OGCG: %[[METHOD_CALL:.*]] = call noundef i32 @_ZZ34cxx_default_init_with_struct_fieldvEN6Parent4getAEv(ptr {{.*}} %[[P_ADDR]])
// OGCG: store i32 %[[METHOD_CALL]], ptr %[[P_ELEM_0_PTR]], align 4
