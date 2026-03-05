// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Bar {
  int a;
  char b;
  void method() {}
  void method2(int a) {}
  int method3(int a) { return a; }
};

struct Foo {
  int a;
  char b;
  Bar z;
};

void baz() {
  Bar b;
  b.method();
  b.method2(4);
  int result = b.method3(4);
  Foo f;
}

struct incomplete;
void yoyo(incomplete *i) {}

//  CHECK-DAG: !rec_incomplete = !cir.record<struct "incomplete" incomplete
//  CHECK-DAG: !rec_Bar = !cir.record<struct "Bar" {!s32i, !s8i}>

//  CHECK-DAG: !rec_Foo = !cir.record<struct "Foo" {!s32i, !s8i, !rec_Bar}>
//  CHECK-DAG: !rec_Mandalore = !cir.record<struct "Mandalore" {!u32i, !cir.ptr<!void>, !s32i} #cir.record.decl.ast>
//  CHECK-DAG: !rec_Adv = !cir.record<class "Adv" {!rec_Mandalore}>
//  CHECK-DAG: !rec_Entry = !cir.record<struct "Entry" {!cir.ptr<!cir.func<(!s32i, !cir.ptr<!s8i>, !cir.ptr<!void>) -> !u32i>>}>

//      CHECK: cir.func {{.*}} @_ZN3Bar6methodEv(%arg0: !cir.ptr<!rec_Bar>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!rec_Bar>, !cir.ptr<!cir.ptr<!rec_Bar>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store{{.*}} %arg0, %0 : !cir.ptr<!rec_Bar>, !cir.ptr<!cir.ptr<!rec_Bar>>
// CHECK-NEXT:   %1 = cir.load{{.*}} %0 : !cir.ptr<!cir.ptr<!rec_Bar>>, !cir.ptr<!rec_Bar>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func {{.*}} @_ZN3Bar7method2Ei(%arg0: !cir.ptr<!rec_Bar> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!rec_Bar>, !cir.ptr<!cir.ptr<!rec_Bar>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store{{.*}} %arg0, %0 : !cir.ptr<!rec_Bar>, !cir.ptr<!cir.ptr<!rec_Bar>>
// CHECK-NEXT:   cir.store{{.*}} %arg1, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %2 = cir.load{{.*}} %0 : !cir.ptr<!cir.ptr<!rec_Bar>>, !cir.ptr<!rec_Bar>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func {{.*}} @_ZN3Bar7method3Ei(%arg0: !cir.ptr<!rec_Bar> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!rec_Bar>, !cir.ptr<!cir.ptr<!rec_Bar>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store{{.*}} %arg0, %0 : !cir.ptr<!rec_Bar>, !cir.ptr<!cir.ptr<!rec_Bar>>
// CHECK-NEXT:   cir.store{{.*}} %arg1, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %3 = cir.load{{.*}} %0 : !cir.ptr<!cir.ptr<!rec_Bar>>, !cir.ptr<!rec_Bar>
// CHECK-NEXT:   %4 = cir.load{{.*}} %1 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.store{{.*}} %4, %2 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %5 = cir.load{{.*}} %2 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.return %5
// CHECK-NEXT: }

//      CHECK: cir.func {{.*}} @_Z3bazv()
// CHECK-NEXT:   %0 = cir.alloca !rec_Bar, !cir.ptr<!rec_Bar>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["result", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !rec_Foo, !cir.ptr<!rec_Foo>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.call @_ZN3Bar6methodEv(%0) : (!cir.ptr<!rec_Bar>) -> ()
// CHECK-NEXT:   %3 = cir.const #cir.int<4> : !s32i
// CHECK-NEXT:   cir.call @_ZN3Bar7method2Ei(%0, %3) : (!cir.ptr<!rec_Bar>, !s32i) -> ()
// CHECK-NEXT:   %4 = cir.const #cir.int<4> : !s32i
// CHECK-NEXT:   %5 = cir.call @_ZN3Bar7method3Ei(%0, %4) : (!cir.ptr<!rec_Bar>, !s32i) -> !s32i
// CHECK-NEXT:   cir.store{{.*}} %5, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

typedef enum Ways {
  ThisIsTheWay = 1000024001,
} Ways;

typedef struct Mandalore {
    Ways             w;
    const void*      n;
    int              d;
} Mandalore;

class Adv {
  Mandalore x{ThisIsTheWay};
public:
  Adv() {}
};

void m() { Adv C; }

// CHECK: cir.func {{.*}} @_ZN3AdvC2Ev(%arg0: !cir.ptr<!rec_Adv>
// CHECK:     %0 = cir.alloca !cir.ptr<!rec_Adv>, !cir.ptr<!cir.ptr<!rec_Adv>>, ["this", init] {alignment = 8 : i64}
// CHECK:     cir.store{{.*}} %arg0, %0 : !cir.ptr<!rec_Adv>, !cir.ptr<!cir.ptr<!rec_Adv>>
// CHECK:     %1 = cir.load{{.*}} %0 : !cir.ptr<!cir.ptr<!rec_Adv>>, !cir.ptr<!rec_Adv>
// CHECK:     %2 = cir.get_member %1[0] {name = "x"} : !cir.ptr<!rec_Adv> -> !cir.ptr<!rec_Mandalore>
// CHECK:     %3 = cir.get_member %2[0] {name = "w"} : !cir.ptr<!rec_Mandalore> -> !cir.ptr<!u32i>
// CHECK:     %4 = cir.const #cir.int<1000024001> : !u32i
// CHECK:     cir.store{{.*}} %4, %3 : !u32i, !cir.ptr<!u32i>
// CHECK:     %5 = cir.get_member %2[1] {name = "n"} : !cir.ptr<!rec_Mandalore> -> !cir.ptr<!cir.ptr<!void>>
// CHECK:     %6 = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CHECK:     cir.store{{.*}} %6, %5 : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CHECK:     %7 = cir.get_member %2[2] {name = "d"} : !cir.ptr<!rec_Mandalore> -> !cir.ptr<!s32i>
// CHECK:     %8 = cir.const #cir.int<0> : !s32i
// CHECK:     cir.store{{.*}} %8, %7 : !s32i, !cir.ptr<!s32i>
// CHECK:     cir.return
// CHECK:   }

struct A {
  int a;
};

// Should globally const-initialize struct members.
struct A simpleConstInit = {1};
// CHECK: cir.global external @simpleConstInit = #cir.const_record<{#cir.int<1> : !s32i}> : !rec_A

// Should globally const-initialize arrays with struct members.
struct A arrConstInit[1] = {{1}};
// CHECK: cir.global external @arrConstInit = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i}> : !rec_A]> : !cir.array<!rec_A x 1>

// Should globally const-initialize empty structs with a non-trivial constexpr
// constructor (as undef, to match existing clang CodeGen behavior).
struct NonTrivialConstexprConstructor {
  constexpr NonTrivialConstexprConstructor() {}
} nonTrivialConstexprConstructor;
// CHECK: cir.global external @nonTrivialConstexprConstructor = #cir.undef : !rec_NonTrivialConstexprConstructor {alignment = 1 : i64}
// CHECK-NOT: @__cxx_global_var_init

// Should locally copy struct members.
void shouldLocallyCopyStructAssignments(void) {
  struct A a = { 3 };
  // CHECK: %[[#SA:]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a", init] {alignment = 4 : i64}
  struct A b = a;
  // CHECK: %[[#SB:]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["b", init] {alignment = 4 : i64}
  // cir.copy %[[#SA]] to %[[SB]] : !cir.ptr<!rec_A>
}

A get_default() { return A{2}; }

struct S {
  S(A a = get_default());
};

void h() { S s; }

// CHECK: cir.func {{.*}} @_Z1hv()
// CHECK:   %0 = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init] {alignment = 1 : i64}
// CHECK:   %1 = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["agg.tmp0"] {alignment = 4 : i64}
// CHECK:   %2 = cir.call @_Z11get_defaultv() : () -> !rec_A
// CHECK:   cir.store{{.*}} %2, %1 : !rec_A, !cir.ptr<!rec_A>
// CHECK:   %3 = cir.load{{.*}} %1 : !cir.ptr<!rec_A>, !rec_A
// CHECK:   cir.call @_ZN1SC1E1A(%0, %3) : (!cir.ptr<!rec_S>, !rec_A) -> ()
// CHECK:   cir.return
// CHECK: }

typedef enum enumy {
  A = 1
} enumy;

typedef enumy (*fnPtr)(int instance, const char* name, void* function);

struct Entry {
  fnPtr procAddr = nullptr;
};

void ppp() { Entry x; }

// CHECK: cir.func {{.*}} @_ZN5EntryC2Ev(%arg0: !cir.ptr<!rec_Entry>

// CHECK: cir.get_member %1[0] {name = "procAddr"} : !cir.ptr<!rec_Entry> -> !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.ptr<!s8i>, !cir.ptr<!void>) -> !u32i>>>

struct CompleteS {
  int a;
  char b;
};

void designated_init_update_expr() {
  CompleteS a;

  struct Container {
    CompleteS c;
  } b = {a, .c.a = 1};
}

// CHECK: %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a"]
// CHECK: %[[B_ADDR:.*]] = cir.alloca !rec_Container, !cir.ptr<!rec_Container>, ["b", init]
// CHECK: %[[C_ADDR:.*]] = cir.get_member %[[B_ADDR]][0] {name = "c"} : !cir.ptr<!rec_Container> -> !cir.ptr<!rec_CompleteS>
// CHECK: cir.copy %[[A_ADDR]] to %[[C_ADDR]] : !cir.ptr<!rec_CompleteS>
// CHECK: %[[ELEM_0_PTR:.*]] = cir.get_member %[[C_ADDR]][0] {name = "a"} : !cir.ptr<!rec_CompleteS> -> !cir.ptr<!s32i>
// CHECK: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CHECK: cir.store{{.*}} %[[CONST_1]], %[[ELEM_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK: %[[ELEM_1_PTR:.*]] = cir.get_member %[[C_ADDR]][1] {name = "b"} : !cir.ptr<!rec_CompleteS> -> !cir.ptr<!s8i>

void unary_extension() {
  CompleteS a = __extension__ CompleteS();
}

// CHECK: %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a", init]
// CHECK: %[[ZERO_INIT:.*]] = cir.const #cir.zero : !rec_CompleteS
// CHECK: cir.store{{.*}} %[[ZERO_INIT]], %[[A_ADDR]] : !rec_CompleteS, !cir.ptr<!rec_CompleteS>

void generic_selection() {
  CompleteS a;
  CompleteS b;
  int c;
  CompleteS d = _Generic(c, int : a, default: b);
}

// CHECK: %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a"]
// CHECK: %[[B_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["b"]
// CHECK: %[[C_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c"]
// CHECK: %[[D_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["d", init]
// CHECK: cir.copy %[[A_ADDR]] to %[[D_ADDR]] : !cir.ptr<!rec_CompleteS>

void choose_expr() {
  CompleteS a;
  CompleteS b;
  CompleteS c = __builtin_choose_expr(true, a, b);
}

// CHECK: cir.func{{.*}} @_Z11choose_exprv()
// CHECK:   %[[A_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["a"]
// CHECK:   %[[B_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["b"]
// CHECK:   %[[C_ADDR:.*]] = cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["c", init]
// CHECK:   cir.copy %[[A_ADDR]] to %[[C_ADDR]] : !cir.ptr<!rec_CompleteS>
