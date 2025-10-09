// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct IncompleteS;
IncompleteS *p;

// CIR: cir.global external @p = #cir.ptr<null> : !cir.ptr<!rec_IncompleteS>
// LLVM: @p = global ptr null
// OGCG: @p = global ptr null, align 8

struct CompleteS {
  int a;
  char b;
};

CompleteS cs;

// CIR:       cir.global external @cs = #cir.zero : !rec_CompleteS
// LLVM-DAG:  @cs = global %struct.CompleteS zeroinitializer
// OGCG-DAG:  @cs = global %struct.CompleteS zeroinitializer, align 4

void f(void) {
  IncompleteS *p;
}

// CIR:      cir.func{{.*}} @_Z1fv()
// CIR-NEXT:   cir.alloca !cir.ptr<!rec_IncompleteS>, !cir.ptr<!cir.ptr<!rec_IncompleteS>>, ["p"]
// CIR-NEXT:   cir.return

// LLVM:      define{{.*}} void @_Z1fv()
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @_Z1fv()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   ret void

char f2(CompleteS &s) {
  return s.b;
}

// CIR: cir.func{{.*}} @_Z2f2R9CompleteS(%[[ARG_S:.*]]: !cir.ptr<!rec_CompleteS>{{.*}})
// CIR:   %[[S_ADDR:.*]] = cir.alloca !cir.ptr<!rec_CompleteS>, !cir.ptr<!cir.ptr<!rec_CompleteS>>, ["s", init, const]
// CIR:   cir.store %[[ARG_S]], %[[S_ADDR]]
// CIR:   %[[S_REF:.*]] = cir.load{{.*}} %[[S_ADDR]]
// CIR:   %[[S_ADDR2:.*]] = cir.get_member %[[S_REF]][1] {name = "b"}
// CIR:   %[[S_B:.*]] = cir.load{{.*}} %[[S_ADDR2]]

// LLVM: define{{.*}} i8 @_Z2f2R9CompleteS(ptr %[[ARG_S:.*]])
// LLVM:   %[[S_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG_S]], ptr %[[S_ADDR]]
// LLVM:   %[[S_REF:.*]] = load ptr, ptr %[[S_ADDR]], align 8
// LLVM:   %[[S_ADDR2:.*]] = getelementptr %struct.CompleteS, ptr %[[S_REF]], i32 0, i32 1
// LLVM:   %[[S_B:.*]] = load i8, ptr %[[S_ADDR2]]

// OGCG: define{{.*}} i8 @_Z2f2R9CompleteS(ptr{{.*}} %[[ARG_S:.*]])
// OGCG: entry:
// OGCG:   %[[S_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG_S]], ptr %[[S_ADDR]]
// OGCG:   %[[S_REF:.*]] = load ptr, ptr %[[S_ADDR]]
// OGCG:   %[[S_ADDR2:.*]] = getelementptr inbounds nuw %struct.CompleteS, ptr %[[S_REF]], i32 0, i32 1
// OGCG:   %[[S_B:.*]] = load i8, ptr %[[S_ADDR2]]

struct Inner {
  int n;
};

struct Outer {
  Inner i;
};

void f3() {
  Outer o;
  o.i.n;
}

// CIR: cir.func{{.*}} @_Z2f3v()
// CIR:   %[[O:.*]] = cir.alloca !rec_Outer, !cir.ptr<!rec_Outer>, ["o"]
// CIR:   %[[O_I:.*]] = cir.get_member %[[O]][0] {name = "i"}
// CIR:   %[[O_I_N:.*]] = cir.get_member %[[O_I]][0] {name = "n"}

// LLVM: define{{.*}} void @_Z2f3v()
// LLVM:   %[[O:.*]] = alloca %struct.Outer, i64 1, align 4
// LLVM:   %[[O_I:.*]] = getelementptr %struct.Outer, ptr %[[O]], i32 0, i32 0
// LLVM:   %[[O_I_N:.*]] = getelementptr %struct.Inner, ptr %[[O_I]], i32 0, i32 0

// OGCG: define{{.*}} void @_Z2f3v()
// OGCG:   %[[O:.*]] = alloca %struct.Outer, align 4
// OGCG:   %[[O_I:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %[[O]], i32 0, i32 0
// OGCG:   %[[O_I_N:.*]] = getelementptr inbounds nuw %struct.Inner, ptr %[[O_I]], i32 0, i32 0

void paren_expr() {
  struct Point {
    int x;
    int y;
  };

  Point a = (Point{});
  Point b = (a);
}

// CIR: cir.func{{.*}} @_Z10paren_exprv()
// CIR:   %[[A_ADDR:.*]] = cir.alloca !rec_Point, !cir.ptr<!rec_Point>, ["a", init]
// CIR:   %[[B_ADDR:.*]] = cir.alloca !rec_Point, !cir.ptr<!rec_Point>, ["b", init]
// CIR:   %[[X_ELEM_PTR:.*]] = cir.get_member %[[A_ADDR]][0] {name = "x"} : !cir.ptr<!rec_Point> -> !cir.ptr<!s32i>
// CIR:   %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[CONST_0]], %[[X_ELEM_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[Y_ELEM_PTR:.*]] = cir.get_member %[[A_ADDR]][1] {name = "y"} : !cir.ptr<!rec_Point> -> !cir.ptr<!s32i>
// CIR:   %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[CONST_0]], %[[Y_ELEM_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.call @_ZZ10paren_exprvEN5PointC1ERKS_(%[[B_ADDR]], %[[A_ADDR]]) nothrow : (!cir.ptr<!rec_Point>, !cir.ptr<!rec_Point>) -> ()

// LLVM: define{{.*}} void @_Z10paren_exprv()
// LLVM:   %[[A_ADDR:.*]] = alloca %struct.Point, i64 1, align 4
// LLVM:   %[[B_ADDR:.*]] = alloca %struct.Point, i64 1, align 4
// LLVM:   %[[X_ELEM_PTR:.*]] = getelementptr %struct.Point, ptr %[[A_ADDR]], i32 0, i32 0
// LLVM:   store i32 0, ptr %[[X_ELEM_PTR]], align 4
// LLVM:   %[[Y_ELEM_PTR:.*]] = getelementptr %struct.Point, ptr %[[A_ADDR]], i32 0, i32 1
// LLVM:   store i32 0, ptr %[[Y_ELEM_PTR]], align 4
// LLVM:   call void @_ZZ10paren_exprvEN5PointC1ERKS_(ptr %[[B_ADDR]], ptr %[[A_ADDR]])

// OGCG: define{{.*}} void @_Z10paren_exprv()
// OGCG:   %[[A_ADDR:.*]] = alloca %struct.Point, align 4
// OGCG:   %[[B_ADDR:.*]] = alloca %struct.Point, align 4
// OGCG:   call void @llvm.memset.p0.i64(ptr align 4 %[[A_ADDR]], i8 0, i64 8, i1 false)
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[B_ADDR]], ptr align 4 %[[A_ADDR]], i64 8, i1 false)

void choose_expr() {
  CompleteS a;
  CompleteS b;
  CompleteS c = __builtin_choose_expr(true, a, b);
}

// CIR: cir.func{{.*}} @_Z11choose_exprv()
// CIR:   %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a"]
// CIR:   %[[B_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["b"]
// CIR:   %[[C_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["c", init]
// TODO(cir): Call to default copy constructor should be replaced by `cir.copy` op
// CIR:   cir.call @_ZN9CompleteSC1ERKS_(%[[C_ADDR]], %[[A_ADDR]]) nothrow : (!cir.ptr<!rec_CompleteS>, !cir.ptr<!rec_CompleteS>) -> ()

// LLVM: define{{.*}} void @_Z11choose_exprv()
// LLVM:   %[[A_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM:   %[[B_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM:   %[[C_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM:   call void @_ZN9CompleteSC1ERKS_(ptr %[[C_ADDR]], ptr %[[A_ADDR]])

// OGCG: define{{.*}} void @_Z11choose_exprv()
// OGCG:   %[[A_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG:   %[[B_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG:   %[[C_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[C_ADDR]], ptr align 4 %[[A_ADDR]], i64 8, i1 false)

void generic_selection() {
  CompleteS a;
  CompleteS b;
  int c;
  CompleteS d = _Generic(c, int : a, default: b);
}

// CIR: cir.func{{.*}} @_Z17generic_selectionv()
// CIR:   %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a"]
// CIR:   %[[B_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["b"]
// CIR:   %[[C_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c"]
// CIR:   %[[D_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["d", init]
// TODO(cir): Call to default copy constructor should be replaced by `cir.copy` op
// CIR:   cir.call @_ZN9CompleteSC1ERKS_(%[[D_ADDR]], %[[A_ADDR]]) nothrow : (!cir.ptr<!rec_CompleteS>, !cir.ptr<!rec_CompleteS>) -> ()

// LLVM: define{{.*}} void @_Z17generic_selectionv()
// LLVM:   %1 = alloca %struct.CompleteS, i64 1, align 4
// LLVM:   %2 = alloca %struct.CompleteS, i64 1, align 4
// LLVM:   %3 = alloca i32, i64 1, align 4
// LLVM:   %4 = alloca %struct.CompleteS, i64 1, align 4
// LLVM:   call void @_ZN9CompleteSC1ERKS_(ptr %4, ptr %1)

// OGCG: define{{.*}} void @_Z17generic_selectionv()
// OGCG:   %[[A_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG:   %[[B_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG:   %[[C_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[D_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[D_ADDR]], ptr align 4 %[[A_ADDR]], i64 8, i1 false)

void designated_init_update_expr() {
  CompleteS a;

  struct Container {
    CompleteS c;
  } b = {a, .c.a = 1};
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !rec_Container, !cir.ptr<!rec_Container>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.get_member %[[B_ADDR]][0] {name = "c"} : !cir.ptr<!rec_Container> -> !cir.ptr<!rec_CompleteS>
// CIR: cir.call @_ZN9CompleteSC1ERKS_(%2, %[[A_ADDR]]) nothrow : (!cir.ptr<!rec_CompleteS>, !cir.ptr<!rec_CompleteS>) -> ()
// CIR: %[[ELEM_0_PTR:.*]] = cir.get_member %[[C_ADDR]][0] {name = "a"} : !cir.ptr<!rec_CompleteS> -> !cir.ptr<!s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST_1]], %[[ELEM_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[ELEM_1_PTR:.*]] = cir.get_member %[[C_ADDR]][1] {name = "b"} : !cir.ptr<!rec_CompleteS> -> !cir.ptr<!s8i>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca %struct.Container, i64 1, align 4
// LLVM: %[[C_ADDR:.*]] = getelementptr %struct.Container, ptr %[[B_ADDR]], i32 0, i32 0
// LLVM: call void @_ZN9CompleteSC1ERKS_(ptr %[[C_ADDR]], ptr %[[A_ADDR]])
// LLVM: %[[ELEM_0_PTR:.*]] = getelementptr %struct.CompleteS, ptr %[[C_ADDR]], i32 0, i32 0
// LLVM: store i32 1, ptr %[[ELEM_0_PTR]], align 4
// LLVM: %[[ELEM_1_PTR:.*]] = getelementptr %struct.CompleteS, ptr %[[C_ADDR]], i32 0, i32 1

// OGCG: %[[A_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG: %[[B_ADDR:.*]] = alloca %struct.Container, align 4
// OGCG: %[[C_ADDR:.*]] = getelementptr inbounds nuw %struct.Container, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[C_ADDR]], ptr align 4 %[[A_ADDR]], i64 8, i1 false)
// OGCG: %[[ELEM_0_PTR:.*]] = getelementptr inbounds nuw %struct.CompleteS, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG: store i32 1, ptr %[[ELEM_0_PTR]], align 4
// OGCG: %[[ELEM_1_PTR:.*]] = getelementptr inbounds nuw %struct.CompleteS, ptr %[[C_ADDR]], i32 0, i32 1

void atomic_init() {
  _Atomic CompleteS a;
  __c11_atomic_init(&a, {});
}

// CIR: cir.func{{.*}} @_Z11atomic_initv()
// CIR:   %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a"]
// CIR:   %[[ELEM_0_PTR:.*]] = cir.get_member %[[A_ADDR]][0] {name = "a"} : !cir.ptr<!rec_CompleteS> -> !cir.ptr<!s32i>
// CIR:   %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[CONST_0]], %[[ELEM_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ELEM_1_PTR:.*]] = cir.get_member %[[A_ADDR]][1] {name = "b"} : !cir.ptr<!rec_CompleteS> -> !cir.ptr<!s8i>
// CIR:   %[[CONST_0:.*]] = cir.const #cir.int<0> : !s8i
// CIR:   cir.store{{.*}} %[[CONST_0]], %[[ELEM_1_PTR]] : !s8i, !cir.ptr<!s8i>

// LLVM: define{{.*}} void @_Z11atomic_initv()
// LLVM:   %[[A_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 8
// LLVM:   %[[ELEM_0_PTR:.*]] = getelementptr %struct.CompleteS, ptr %[[A_ADDR]], i32 0, i32 0
// LLVM:   store i32 0, ptr %[[ELEM_0_PTR]], align 8
// LLVM:   %[[ELEM_1_PTR:.*]] = getelementptr %struct.CompleteS, ptr %[[A_ADDR]], i32 0, i32 1
// LLVM:   store i8 0, ptr %[[ELEM_1_PTR]], align 4

// OGCG: define{{.*}} void @_Z11atomic_initv()
// OGCG:   %[[A_ADDR:.*]] = alloca %struct.CompleteS, align 8
// OGCG:   %[[ELEM_0_PTR:.*]] = getelementptr inbounds nuw %struct.CompleteS, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG:   store i32 0, ptr %[[ELEM_0_PTR]], align 8
// OGCG:   %[[ELEM_1_PTR:.*]] = getelementptr inbounds nuw %struct.CompleteS, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG:   store i8 0, ptr %[[ELEM_1_PTR]], align 4

void unary_extension() {
  CompleteS a = __extension__ CompleteS();
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a", init]
// CIR: %[[ZERO_INIT:.*]] = cir.const #cir.zero : !rec_CompleteS
// CIR: cir.store{{.*}} %[[ZERO_INIT]], %[[A_ADDR]] : !rec_CompleteS, !cir.ptr<!rec_CompleteS>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM: store %struct.CompleteS zeroinitializer, ptr %[[A_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG: call void @llvm.memset.p0.i64(ptr align 4 %[[A_ADDR]], i8 0, i64 8, i1 false)

void bin_comma() { 
  CompleteS a = (CompleteS(), CompleteS());
}

// CIR: cir.func{{.*}} @_Z9bin_commav()
// CIR:   %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a", init]
// CIR:   %[[TMP_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["agg.tmp0"]
// CIR:   %[[ZERO:.*]] = cir.const #cir.zero : !rec_CompleteS
// CIR:   cir.store{{.*}} %[[ZERO]], %[[TMP_ADDR]] : !rec_CompleteS, !cir.ptr<!rec_CompleteS>
// CIR:   %[[ZERO:.*]] = cir.const #cir.zero : !rec_CompleteS
// CIR:   cir.store{{.*}} %[[ZERO]], %[[A_ADDR]] : !rec_CompleteS, !cir.ptr<!rec_CompleteS>

// LLVM: define{{.*}} void @_Z9bin_commav()
// LLVM:   %[[A_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM:   %[[TMP_ADDR:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM:   store %struct.CompleteS zeroinitializer, ptr %[[TMP_ADDR]], align 4
// LLVM:   store %struct.CompleteS zeroinitializer, ptr %[[A_ADDR]], align 4

// OGCG: define{{.*}} void @_Z9bin_commav()
// OGCG:   %[[A_ADDR:.*]] = alloca %struct.CompleteS, align 4
// OGCG:   call void @llvm.memset.p0.i64(ptr align 4 %[[A_ADDR]], i8 0, i64 8, i1 false)
