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

//  CHECK-DAG-DAG: !ty_22incomplete22 = !cir.struct<struct "incomplete" incomplete
//  CHECK-DAG: !ty_22Bar22 = !cir.struct<struct "Bar" {!s32i, !s8i}>

//  CHECK-DAG: !ty_22Foo22 = !cir.struct<struct "Foo" {!s32i, !s8i, !ty_22Bar22}>
//  CHECK-DAG: !ty_22Mandalore22 = !cir.struct<struct "Mandalore" {!u32i, !cir.ptr<!void>, !s32i} #cir.recdecl.ast>
//  CHECK-DAG: !ty_22Adv22 = !cir.struct<class "Adv" {!ty_22Mandalore22}>
//  CHECK-DAG: !ty_22Entry22 = !cir.struct<struct "Entry" {!cir.ptr<!cir.func<!u32i (!s32i, !cir.ptr<!s8i>, !cir.ptr<!void>)>>}>

//      CHECK: cir.func linkonce_odr @_ZN3Bar6methodEv(%arg0: !cir.ptr<!ty_22Bar22>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Bar22>, cir.ptr <!cir.ptr<!ty_22Bar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Bar22>, cir.ptr <!cir.ptr<!ty_22Bar22>>
// CHECK-NEXT:   %1 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Bar22>>, !cir.ptr<!ty_22Bar22>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method2Ei(%arg0: !cir.ptr<!ty_22Bar22> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Bar22>, cir.ptr <!cir.ptr<!ty_22Bar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Bar22>, cir.ptr <!cir.ptr<!ty_22Bar22>>
// CHECK-NEXT:   cir.store %arg1, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Bar22>>, !cir.ptr<!ty_22Bar22>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method3Ei(%arg0: !cir.ptr<!ty_22Bar22> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Bar22>, cir.ptr <!cir.ptr<!ty_22Bar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Bar22>, cir.ptr <!cir.ptr<!ty_22Bar22>>
// CHECK-NEXT:   cir.store %arg1, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %3 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Bar22>>, !cir.ptr<!ty_22Bar22>
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.store %4, %2 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %5 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.return %5
// CHECK-NEXT: }

//      CHECK: cir.func @_Z3bazv()
// CHECK-NEXT:   %0 = cir.alloca !ty_22Bar22, cir.ptr <!ty_22Bar22>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["result", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !ty_22Foo22, cir.ptr <!ty_22Foo22>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.call @_ZN3Bar6methodEv(%0) : (!cir.ptr<!ty_22Bar22>) -> ()
// CHECK-NEXT:   %3 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK-NEXT:   cir.call @_ZN3Bar7method2Ei(%0, %3) : (!cir.ptr<!ty_22Bar22>, !s32i) -> ()
// CHECK-NEXT:   %4 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK-NEXT:   %5 = cir.call @_ZN3Bar7method3Ei(%0, %4) : (!cir.ptr<!ty_22Bar22>, !s32i) -> !s32i
// CHECK-NEXT:   cir.store %5, %1 : !s32i, cir.ptr <!s32i>
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

// CHECK: cir.func linkonce_odr @_ZN3AdvC2Ev(%arg0: !cir.ptr<!ty_22Adv22>
// CHECK:     %0 = cir.alloca !cir.ptr<!ty_22Adv22>, cir.ptr <!cir.ptr<!ty_22Adv22>>, ["this", init] {alignment = 8 : i64}
// CHECK:     cir.store %arg0, %0 : !cir.ptr<!ty_22Adv22>, cir.ptr <!cir.ptr<!ty_22Adv22>>
// CHECK:     %1 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Adv22>>, !cir.ptr<!ty_22Adv22>
// CHECK:     %2 = "cir.struct_element_addr"(%1) <{member_index = 0 : index, member_name = "x"}> : (!cir.ptr<!ty_22Adv22>) -> !cir.ptr<!ty_22Mandalore22>
// CHECK:     %3 = "cir.struct_element_addr"(%2) <{member_index = 0 : index, member_name = "w"}> : (!cir.ptr<!ty_22Mandalore22>) -> !cir.ptr<!u32i>
// CHECK:     %4 = cir.const(#cir.int<1000024001> : !u32i) : !u32i
// CHECK:     cir.store %4, %3 : !u32i, cir.ptr <!u32i>
// CHECK:     %5 = "cir.struct_element_addr"(%2) <{member_index = 1 : index, member_name = "n"}> : (!cir.ptr<!ty_22Mandalore22>) -> !cir.ptr<!cir.ptr<!void>>
// CHECK:     %6 = cir.const(#cir.null : !cir.ptr<!void>) : !cir.ptr<!void>
// CHECK:     cir.store %6, %5 : !cir.ptr<!void>, cir.ptr <!cir.ptr<!void>>
// CHECK:     %7 = "cir.struct_element_addr"(%2) <{member_index = 2 : index, member_name = "d"}> : (!cir.ptr<!ty_22Mandalore22>) -> !cir.ptr<!s32i>
// CHECK:     %8 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:     cir.store %8, %7 : !s32i, cir.ptr <!s32i>
// CHECK:     cir.return
// CHECK:   }

struct A {
  int a;
};

// Should globally const-initialize struct members.
struct A simpleConstInit = {1};
// CHECK: cir.global external @simpleConstInit = #cir.const_struct<{#cir.int<1> : !s32i}> : !ty_22A22

// Should globally const-initialize arrays with struct members.
struct A arrConstInit[1] = {{1}};
// CHECK: cir.global external @arrConstInit = #cir.const_array<[#cir.const_struct<{#cir.int<1> : !s32i}> : !ty_22A22]> : !cir.array<!ty_22A22 x 1>

// Should locally copy struct members.
void shouldLocallyCopyStructAssignments(void) {
  struct A a = { 3 };
  // CHECK: %[[#SA:]] = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["a"] {alignment = 4 : i64}
  struct A b = a;
  // CHECK: %[[#SB:]] = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["b", init] {alignment = 4 : i64}
  // cir.copy %[[#SA]] to %[[SB]] : !cir.ptr<!ty_22A22>
}

A get_default() { return A{2}; }

struct S {
  S(A a = get_default());
};

void h() { S s; }

// CHECK: cir.func @_Z1hv()
// CHECK:   %0 = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>, ["s", init] {alignment = 1 : i64}
// CHECK:   %1 = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["agg.tmp0"] {alignment = 4 : i64}
// CHECK:   %2 = cir.call @_Z11get_defaultv() : () -> !ty_22A22
// CHECK:   cir.store %2, %1 : !ty_22A22, cir.ptr <!ty_22A22>
// CHECK:   %3 = cir.load %1 : cir.ptr <!ty_22A22>, !ty_22A22
// CHECK:   cir.call @_ZN1SC1E1A(%0, %3) : (!cir.ptr<!ty_22S22>, !ty_22A22) -> ()
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

// CHECK: cir.func linkonce_odr @_ZN5EntryC2Ev(%arg0: !cir.ptr<!ty_22Entry22>

// CHECK: = "cir.struct_element_addr"(%1) <{member_index = 0 : index, member_name = "procAddr"}> : (!cir.ptr<!ty_22Entry22>) -> !cir.ptr<!cir.ptr<!cir.func<!u32i (!s32i, !cir.ptr<!s8i>, !cir.ptr<!void>)>>>
