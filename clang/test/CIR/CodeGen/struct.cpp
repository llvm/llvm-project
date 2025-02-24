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

//  CHECK-DAG: !ty_incomplete = !cir.struct<struct "incomplete" incomplete
//  CHECK-DAG: !ty_Bar = !cir.struct<struct "Bar" {!s32i, !s8i}>

//  CHECK-DAG: !ty_Foo = !cir.struct<struct "Foo" {!s32i, !s8i, !ty_Bar}>
//  CHECK-DAG: !ty_Mandalore = !cir.struct<struct "Mandalore" {!u32i, !cir.ptr<!void>, !s32i} #cir.record.decl.ast>
//  CHECK-DAG: !ty_Adv = !cir.struct<class "Adv" {!ty_Mandalore}>
//  CHECK-DAG: !ty_Entry = !cir.struct<struct "Entry" {!cir.ptr<!cir.func<(!s32i, !cir.ptr<!s8i>, !cir.ptr<!void>) -> !u32i>>}>

//      CHECK: cir.func linkonce_odr @_ZN3Bar6methodEv(%arg0: !cir.ptr<!ty_Bar>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_Bar>, !cir.ptr<!cir.ptr<!ty_Bar>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_Bar>, !cir.ptr<!cir.ptr<!ty_Bar>>
// CHECK-NEXT:   %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_Bar>>, !cir.ptr<!ty_Bar>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method2Ei(%arg0: !cir.ptr<!ty_Bar> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_Bar>, !cir.ptr<!cir.ptr<!ty_Bar>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_Bar>, !cir.ptr<!cir.ptr<!ty_Bar>>
// CHECK-NEXT:   cir.store %arg1, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %2 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_Bar>>, !cir.ptr<!ty_Bar>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method3Ei(%arg0: !cir.ptr<!ty_Bar> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_Bar>, !cir.ptr<!cir.ptr<!ty_Bar>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_Bar>, !cir.ptr<!cir.ptr<!ty_Bar>>
// CHECK-NEXT:   cir.store %arg1, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %3 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_Bar>>, !cir.ptr<!ty_Bar>
// CHECK-NEXT:   %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.store %4, %2 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.return %5
// CHECK-NEXT: }

//      CHECK: cir.func @_Z3bazv()
// CHECK-NEXT:   %0 = cir.alloca !ty_Bar, !cir.ptr<!ty_Bar>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["result", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !ty_Foo, !cir.ptr<!ty_Foo>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.call @_ZN3Bar6methodEv(%0) : (!cir.ptr<!ty_Bar>) -> ()
// CHECK-NEXT:   %3 = cir.const #cir.int<4> : !s32i
// CHECK-NEXT:   cir.call @_ZN3Bar7method2Ei(%0, %3) : (!cir.ptr<!ty_Bar>, !s32i) -> ()
// CHECK-NEXT:   %4 = cir.const #cir.int<4> : !s32i
// CHECK-NEXT:   %5 = cir.call @_ZN3Bar7method3Ei(%0, %4) : (!cir.ptr<!ty_Bar>, !s32i) -> !s32i
// CHECK-NEXT:   cir.store %5, %1 : !s32i, !cir.ptr<!s32i>
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

// CHECK: cir.func linkonce_odr @_ZN3AdvC2Ev(%arg0: !cir.ptr<!ty_Adv>
// CHECK:     %0 = cir.alloca !cir.ptr<!ty_Adv>, !cir.ptr<!cir.ptr<!ty_Adv>>, ["this", init] {alignment = 8 : i64}
// CHECK:     cir.store %arg0, %0 : !cir.ptr<!ty_Adv>, !cir.ptr<!cir.ptr<!ty_Adv>>
// CHECK:     %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_Adv>>, !cir.ptr<!ty_Adv>
// CHECK:     %2 = cir.get_member %1[0] {name = "x"} : !cir.ptr<!ty_Adv> -> !cir.ptr<!ty_Mandalore>
// CHECK:     %3 = cir.get_member %2[0] {name = "w"} : !cir.ptr<!ty_Mandalore> -> !cir.ptr<!u32i>
// CHECK:     %4 = cir.const #cir.int<1000024001> : !u32i
// CHECK:     cir.store %4, %3 : !u32i, !cir.ptr<!u32i>
// CHECK:     %5 = cir.get_member %2[1] {name = "n"} : !cir.ptr<!ty_Mandalore> -> !cir.ptr<!cir.ptr<!void>>
// CHECK:     %6 = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CHECK:     cir.store %6, %5 : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CHECK:     %7 = cir.get_member %2[2] {name = "d"} : !cir.ptr<!ty_Mandalore> -> !cir.ptr<!s32i>
// CHECK:     %8 = cir.const #cir.int<0> : !s32i
// CHECK:     cir.store %8, %7 : !s32i, !cir.ptr<!s32i>
// CHECK:     cir.return
// CHECK:   }

struct A {
  int a;
};

// Should globally const-initialize struct members.
struct A simpleConstInit = {1};
// CHECK: cir.global external @simpleConstInit = #cir.const_struct<{#cir.int<1> : !s32i}> : !ty_A

// Should globally const-initialize arrays with struct members.
struct A arrConstInit[1] = {{1}};
// CHECK: cir.global external @arrConstInit = #cir.const_array<[#cir.const_struct<{#cir.int<1> : !s32i}> : !ty_A]> : !cir.array<!ty_A x 1>

// Should globally const-initialize empty structs with a non-trivial constexpr
// constructor (as undef, to match existing clang CodeGen behavior).
struct NonTrivialConstexprConstructor {
  constexpr NonTrivialConstexprConstructor() {}
} nonTrivialConstexprConstructor;
// CHECK: cir.global external @nonTrivialConstexprConstructor = #cir.undef : !ty_NonTrivialConstexprConstructor {alignment = 1 : i64}
// CHECK-NOT: @__cxx_global_var_init

// Should locally copy struct members.
void shouldLocallyCopyStructAssignments(void) {
  struct A a = { 3 };
  // CHECK: %[[#SA:]] = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["a"] {alignment = 4 : i64}
  struct A b = a;
  // CHECK: %[[#SB:]] = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["b", init] {alignment = 4 : i64}
  // cir.copy %[[#SA]] to %[[SB]] : !cir.ptr<!ty_A>
}

A get_default() { return A{2}; }

struct S {
  S(A a = get_default());
};

void h() { S s; }

// CHECK: cir.func @_Z1hv()
// CHECK:   %0 = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["s", init] {alignment = 1 : i64}
// CHECK:   %1 = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["agg.tmp0"] {alignment = 4 : i64}
// CHECK:   %2 = cir.call @_Z11get_defaultv() : () -> !ty_A
// CHECK:   cir.store %2, %1 : !ty_A, !cir.ptr<!ty_A>
// CHECK:   %3 = cir.load %1 : !cir.ptr<!ty_A>, !ty_A
// CHECK:   cir.call @_ZN1SC1E1A(%0, %3) : (!cir.ptr<!ty_S>, !ty_A) -> ()
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

// CHECK: cir.func linkonce_odr @_ZN5EntryC2Ev(%arg0: !cir.ptr<!ty_Entry>

// CHECK: cir.get_member %1[0] {name = "procAddr"} : !cir.ptr<!ty_Entry> -> !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.ptr<!s8i>, !cir.ptr<!void>) -> !u32i>>>
