// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

union U1 {
  int n;
  char c;
};

// CIR:  !rec_U1 = !cir.record<union "U1" {!s32i, !s8i}>
// LLVM: %union.U1 = type { i32 }
// OGCG: %union.U1 = type { i32 }

union U2 {
  char b;
  short s;
  int i;
  float f;
  double d;
};

// CIR:  !rec_U2 = !cir.record<union "U2" {!s8i, !s16i, !s32i, !cir.float, !cir.double}>
// LLVM: %union.U2 = type { double }
// OGCG: %union.U2 = type { double }

union U3 {
  char c[5];
  int i;
} __attribute__((packed));

// CIR:  !rec_U3 = !cir.record<union "U3" packed padded {!cir.array<!s8i x 5>, !s32i, !u8i}>
// LLVM: %union.U3 = type <{ i32, i8 }>
// OGCG: %union.U3 = type <{ i32, i8 }>

union U4 {
  char c[5];
  int i;
};

// CIR:  !rec_U4 = !cir.record<union "U4" padded {!cir.array<!s8i x 5>, !s32i, !cir.array<!u8i x 4>}>
// LLVM: %union.U4 = type { i32, [4 x i8] }
// OGCG: %union.U4 = type { i32, [4 x i8] }

union IncompleteU *p;

// CIR:  cir.global external @p = #cir.ptr<null> : !cir.ptr<!rec_IncompleteU>
// LLVM: @p = dso_local global ptr null
// OGCG: @p = global ptr null, align 8

void f1(void) {
  union IncompleteU *p;
}

// CIR:      cir.func @f1()
// CIR-NEXT:   cir.alloca !cir.ptr<!rec_IncompleteU>, !cir.ptr<!cir.ptr<!rec_IncompleteU>>, ["p"]
// CIR-NEXT:   cir.return

// LLVM:      define void @f1()
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @f1()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   ret void

int f2(void) {
  union U1 u;
  u.n = 42;
  return u.n;
}

// CIR:      cir.func @f2() -> !s32i
// CIR-NEXT:   %[[RETVAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:   %[[U:.*]] = cir.alloca !rec_U1, !cir.ptr<!rec_U1>, ["u"] {alignment = 4 : i64}
// CIR-NEXT:   %[[I:.*]] = cir.const #cir.int<42> : !s32i
// CIR-NEXT:   %[[N:.*]] = cir.get_member %[[U]][0] {name = "n"} : !cir.ptr<!rec_U1> -> !cir.ptr<!s32i>
// CIR-NEXT:   cir.store{{.*}} %[[I]], %[[N]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[N2:.*]] = cir.get_member %[[U]][0] {name = "n"} : !cir.ptr<!rec_U1> -> !cir.ptr<!s32i>
// CIR-NEXT:   %[[VAL:.*]] = cir.load{{.*}} %[[N2]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.store{{.*}} %[[VAL]], %[[RETVAL_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %[[RET]] : !s32i

// LLVM:      define i32 @f2()
// LLVM-NEXT:   %[[RETVAL:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   %[[U:.*]] = alloca %union.U1, i64 1, align 4
// LLVM-NEXT:   store i32 42, ptr %[[U]], align 4
// LLVM-NEXT:   %[[N_VAL:.*]] = load i32, ptr %[[U]], align 4
// LLVM-NEXT:   store i32 %[[N_VAL]], ptr %[[RETVAL]], align 4
// LLVM-NEXT:   %[[RET:.*]] = load i32, ptr %[[RETVAL]], align 4
// LLVM-NEXT:   ret i32 %[[RET]]

//      OGCG: define dso_local i32 @f2()
// OGCG-NEXT: entry:
// OGCG-NEXT: %[[U:.*]] = alloca %union.U1, align 4
// OGCG-NEXT: store i32 42, ptr %[[U]], align 4
// OGCG-NEXT: %[[N_VAL:.*]] = load i32, ptr %[[U]], align 4
// OGCG-NEXT: ret i32 %[[N_VAL]]

void shouldGenerateUnionAccess(union U2 u) {
  u.b = 0;
  u.b;
  u.i = 1;
  u.i;
  u.f = 0.1F;
  u.f;
  u.d = 0.1;
  u.d;
}

// CIR:      cir.func @shouldGenerateUnionAccess(%[[ARG:.*]]: !rec_U2
// CIR-NEXT:   %[[U:.*]] = cir.alloca !rec_U2, !cir.ptr<!rec_U2>, ["u", init] {alignment = 8 : i64}
// CIR-NEXT:   cir.store{{.*}} %[[ARG]], %[[U]] : !rec_U2, !cir.ptr<!rec_U2>
// CIR-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   %[[ZERO_CHAR:.*]] = cir.cast(integral, %[[ZERO]] : !s32i), !s8i
// CIR-NEXT:   %[[B_PTR:.*]] = cir.get_member %[[U]][0] {name = "b"} : !cir.ptr<!rec_U2> -> !cir.ptr<!s8i>
// CIR-NEXT:   cir.store{{.*}} %[[ZERO_CHAR]], %[[B_PTR]] : !s8i, !cir.ptr<!s8i>
// CIR-NEXT:   %[[B_PTR2:.*]] = cir.get_member %[[U]][0] {name = "b"} : !cir.ptr<!rec_U2> -> !cir.ptr<!s8i>
// CIR-NEXT:   %[[B_VAL:.*]] = cir.load{{.*}} %[[B_PTR2]] : !cir.ptr<!s8i>, !s8i
// CIR-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:   %[[I_PTR:.*]] = cir.get_member %[[U]][2] {name = "i"} : !cir.ptr<!rec_U2> -> !cir.ptr<!s32i>
// CIR-NEXT:   cir.store{{.*}} %[[ONE]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[I_PTR2:.*]] = cir.get_member %[[U]][2] {name = "i"} : !cir.ptr<!rec_U2> -> !cir.ptr<!s32i>
// CIR-NEXT:   %[[I_VAL:.*]] = cir.load{{.*}} %[[I_PTR2]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[FLOAT_VAL:.*]] = cir.const #cir.fp<1.000000e-01> : !cir.float
// CIR-NEXT:   %[[F_PTR:.*]] = cir.get_member %[[U]][3] {name = "f"} : !cir.ptr<!rec_U2> -> !cir.ptr<!cir.float>
// CIR-NEXT:   cir.store{{.*}} %[[FLOAT_VAL]], %[[F_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CIR-NEXT:   %[[F_PTR2:.*]] = cir.get_member %[[U]][3] {name = "f"} : !cir.ptr<!rec_U2> -> !cir.ptr<!cir.float>
// CIR-NEXT:   %[[F_VAL:.*]] = cir.load{{.*}} %[[F_PTR2]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT:   %[[DOUBLE_VAL:.*]] = cir.const #cir.fp<1.000000e-01> : !cir.double
// CIR-NEXT:   %[[D_PTR:.*]] = cir.get_member %[[U]][4] {name = "d"} : !cir.ptr<!rec_U2> -> !cir.ptr<!cir.double>
// CIR-NEXT:   cir.store{{.*}} %[[DOUBLE_VAL]], %[[D_PTR]] : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT:   %[[D_PTR2:.*]] = cir.get_member %[[U]][4] {name = "d"} : !cir.ptr<!rec_U2> -> !cir.ptr<!cir.double>
// CIR-NEXT:   %[[D_VAL:.*]] = cir.load{{.*}} %[[D_PTR2]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT:   cir.return

// LLVM:      define void @shouldGenerateUnionAccess(%union.U2 %[[ARG:.*]])
// LLVM-NEXT:   %[[U:.*]] = alloca %union.U2, i64 1, align 8
// LLVM-NEXT:   store %union.U2 %[[ARG]], ptr %[[U]], align 8
// LLVM-NEXT:   store i8 0, ptr %[[U]], align 8
// LLVM-NEXT:   %[[B_VAL:.*]] = load i8, ptr %[[U]], align 8
// LLVM-NEXT:   store i32 1, ptr %[[U]], align 8
// LLVM-NEXT:   %[[I_VAL:.*]] = load i32, ptr %[[U]], align 8
// LLVM-NEXT:   store float 0x3FB99999A0000000, ptr %[[U]], align 8
// LLVM-NEXT:   %[[F_VAL:.*]] = load float, ptr %[[U]], align 8
// LLVM-NEXT:   store double 1.000000e-01, ptr %[[U]], align 8
// LLVM-NEXT:   %[[D_VAL:.*]] = load double, ptr %[[U]], align 8
// LLVM-NEXT:   ret void

// OGCG:      define dso_local void @shouldGenerateUnionAccess(i64 %[[ARG:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[U:.*]] = alloca %union.U2, align 8
// OGCG-NEXT:   %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %union.U2, ptr %[[U]], i32 0, i32 0
// OGCG-NEXT:   store i64 %[[ARG]], ptr %[[COERCE_DIVE]], align 8
// OGCG-NEXT:   store i8 0, ptr %[[U]], align 8
// OGCG-NEXT:   %[[B_VAL:.*]] = load i8, ptr %[[U]], align 8
// OGCG-NEXT:   store i32 1, ptr %[[U]], align 8
// OGCG-NEXT:   %[[I_VAL:.*]] = load i32, ptr %[[U]], align 8
// OGCG-NEXT:   store float 0x3FB99999A0000000, ptr %[[U]], align 8
// OGCG-NEXT:   %[[F_VAL:.*]] = load float, ptr %[[U]], align 8
// OGCG-NEXT:   store double 1.000000e-01, ptr %[[U]], align 8
// OGCG-NEXT:   %[[D_VAL:.*]] = load double, ptr %[[U]], align 8
// OGCG-NEXT:   ret void

void f3(union U3 u) {
  u.c[2] = 0;
}

// CIR:      cir.func @f3(%[[ARG:.*]]: !rec_U3
// CIR-NEXT:   %[[U:.*]] = cir.alloca !rec_U3, !cir.ptr<!rec_U3>, ["u", init] {alignment = 1 : i64}
// CIR-NEXT:   cir.store{{.*}} %[[ARG]], %[[U]] : !rec_U3, !cir.ptr<!rec_U3>
// CIR-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   %[[ZERO_CHAR:.*]] = cir.cast(integral, %[[ZERO]] : !s32i), !s8i
// CIR-NEXT:   %[[IDX:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT:   %[[C_PTR:.*]] = cir.get_member %[[U]][0] {name = "c"} : !cir.ptr<!rec_U3> -> !cir.ptr<!cir.array<!s8i x 5>>
// CIR-NEXT:   %[[C_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[C_PTR]] : !cir.ptr<!cir.array<!s8i x 5>>), !cir.ptr<!s8i>
// CIR-NEXT:   %[[ELEM_PTR:.*]] = cir.ptr_stride(%[[C_DECAY]] : !cir.ptr<!s8i>, %[[IDX]] : !s32i), !cir.ptr<!s8i>
// CIR-NEXT:   cir.store{{.*}} %[[ZERO_CHAR]], %[[ELEM_PTR]] : !s8i, !cir.ptr<!s8i>
// CIR-NEXT:   cir.return

// LLVM:      define void @f3(%union.U3 %[[ARG:.*]])
// LLVM-NEXT:   %[[U:.*]] = alloca %union.U3, i64 1, align 1
// LLVM-NEXT:   store %union.U3 %[[ARG]], ptr %[[U]], align 1
// LLVM-NEXT:   %[[C_PTR:.*]] = getelementptr i8, ptr %[[U]], i32 0
// LLVM-NEXT:   %[[ELEM_PTR:.*]] = getelementptr i8, ptr %[[C_PTR]], i64 2
// LLVM-NEXT:   store i8 0, ptr %[[ELEM_PTR]], align 1
// LLVM-NEXT:   ret void

// OGCG:      define dso_local void @f3(i40 %[[ARG:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[U:.*]] = alloca %union.U3, align 1
// OGCG-NEXT:   store i40 %[[ARG]], ptr %[[U]], align 1
// OGCG-NEXT:   %[[ARRAYIDX:.*]] = getelementptr inbounds [5 x i8], ptr %[[U]], i64 0, i64 2
// OGCG-NEXT:   store i8 0, ptr %[[ARRAYIDX]], align 1
// OGCG-NEXT:   ret void

void f5(union U4 u) {
  u.c[4] = 65;
}

// CIR:      cir.func @f5(%[[ARG:.*]]: !rec_U4
// CIR-NEXT:   %[[U:.*]] = cir.alloca !rec_U4, !cir.ptr<!rec_U4>, ["u", init] {alignment = 4 : i64}
// CIR-NEXT:   cir.store{{.*}} %[[ARG]], %[[U]] : !rec_U4, !cir.ptr<!rec_U4>
// CIR-NEXT:   %[[CHAR_VAL:.*]] = cir.const #cir.int<65> : !s32i
// CIR-NEXT:   %[[CHAR_CAST:.*]] = cir.cast(integral, %[[CHAR_VAL]] : !s32i), !s8i
// CIR-NEXT:   %[[IDX:.*]] = cir.const #cir.int<4> : !s32i
// CIR-NEXT:   %[[C_PTR:.*]] = cir.get_member %[[U]][0] {name = "c"} : !cir.ptr<!rec_U4> -> !cir.ptr<!cir.array<!s8i x 5>>
// CIR-NEXT:   %[[C_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[C_PTR]] : !cir.ptr<!cir.array<!s8i x 5>>), !cir.ptr<!s8i>
// CIR-NEXT:   %[[ELEM_PTR:.*]] = cir.ptr_stride(%[[C_DECAY]] : !cir.ptr<!s8i>, %[[IDX]] : !s32i), !cir.ptr<!s8i>
// CIR-NEXT:   cir.store{{.*}} %[[CHAR_CAST]], %[[ELEM_PTR]] : !s8i, !cir.ptr<!s8i>
// CIR-NEXT:   cir.return

// LLVM:      define void @f5(%union.U4 %[[ARG:.*]])
// LLVM-NEXT:   %[[U:.*]] = alloca %union.U4, i64 1, align 4
// LLVM-NEXT:   store %union.U4 %[[ARG]], ptr %[[U]], align 4
// LLVM-NEXT:   %[[C_PTR:.*]] = getelementptr i8, ptr %[[U]], i32 0
// LLVM-NEXT:   %[[ELEM_PTR:.*]] = getelementptr i8, ptr %[[C_PTR]], i64 4
// LLVM-NEXT:   store i8 65, ptr %[[ELEM_PTR]], align 4
// LLVM-NEXT:   ret void

// OGCG:      define dso_local void @f5(i64 %[[ARG:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[U:.*]] = alloca %union.U4, align 4
// OGCG-NEXT:   store i64 %[[ARG]], ptr %[[U]], align 4
// OGCG-NEXT:   %[[ARRAYIDX:.*]] = getelementptr inbounds [5 x i8], ptr %[[U]], i64 0, i64 4
// OGCG-NEXT:   store i8 65, ptr %[[ARRAYIDX]], align 4
// OGCG-NEXT:   ret void
