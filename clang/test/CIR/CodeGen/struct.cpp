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

//      CHECK: !ty_22struct2Eincomplete22 = !cir.struct<"struct.incomplete", incomplete
//      CHECK: !ty_22struct2EBar22 = !cir.struct<"struct.Bar", !s32i, !s8i>

//      CHECK: !ty_22struct2EFoo22 = !cir.struct<"struct.Foo", !s32i, !s8i, !ty_22struct2EBar22>
//      CHECK: !ty_22struct2EMandalore22 = !cir.struct<"struct.Mandalore", !u32i, !cir.ptr<!void>, !s32i, #cir.recdecl.ast>
//      CHECK: !ty_22class2EAdv22 = !cir.struct<"class.Adv", !ty_22struct2EMandalore22>
//      CHECK: !ty_22struct2EEntry22 = !cir.struct<"struct.Entry", !cir.ptr<!cir.func<!u32i (!s32i, !cir.ptr<!s8i>, !cir.ptr<!void>)>>>

//      CHECK: cir.func linkonce_odr @_ZN3Bar6methodEv(%arg0: !cir.ptr<!ty_22struct2EBar22>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>
// CHECK-NEXT:   %1 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, !cir.ptr<!ty_22struct2EBar22>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method2Ei(%arg0: !cir.ptr<!ty_22struct2EBar22> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>
// CHECK-NEXT:   cir.store %arg1, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, !cir.ptr<!ty_22struct2EBar22>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method3Ei(%arg0: !cir.ptr<!ty_22struct2EBar22> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>
// CHECK-NEXT:   cir.store %arg1, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %3 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, !cir.ptr<!ty_22struct2EBar22>
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.store %4, %2 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %5 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.return %5
// CHECK-NEXT: }

//      CHECK: cir.func @_Z3bazv()
// CHECK-NEXT:   %0 = cir.alloca !ty_22struct2EBar22, cir.ptr <!ty_22struct2EBar22>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["result", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.call @_ZN3Bar6methodEv(%0) : (!cir.ptr<!ty_22struct2EBar22>) -> ()
// CHECK-NEXT:   %3 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK-NEXT:   cir.call @_ZN3Bar7method2Ei(%0, %3) : (!cir.ptr<!ty_22struct2EBar22>, !s32i) -> ()
// CHECK-NEXT:   %4 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK-NEXT:   %5 = cir.call @_ZN3Bar7method3Ei(%0, %4) : (!cir.ptr<!ty_22struct2EBar22>, !s32i) -> !s32i
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

// CHECK: cir.func linkonce_odr @_ZN3AdvC2Ev(%arg0: !cir.ptr<!ty_22class2EAdv22>
// CHECK:     %0 = cir.alloca !cir.ptr<!ty_22class2EAdv22>, cir.ptr <!cir.ptr<!ty_22class2EAdv22>>, ["this", init] {alignment = 8 : i64}
// CHECK:     cir.store %arg0, %0 : !cir.ptr<!ty_22class2EAdv22>, cir.ptr <!cir.ptr<!ty_22class2EAdv22>>
// CHECK:     %1 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22class2EAdv22>>, !cir.ptr<!ty_22class2EAdv22>
// CHECK:     %2 = "cir.struct_element_addr"(%1) <{member_index = 0 : index, member_name = "x"}> : (!cir.ptr<!ty_22class2EAdv22>) -> !cir.ptr<!ty_22struct2EMandalore22>
// CHECK:     %3 = "cir.struct_element_addr"(%2) <{member_index = 0 : index, member_name = "w"}> : (!cir.ptr<!ty_22struct2EMandalore22>) -> !cir.ptr<!u32i>
// CHECK:     %4 = cir.const(#cir.int<1000024001> : !u32i) : !u32i
// CHECK:     cir.store %4, %3 : !u32i, cir.ptr <!u32i>
// CHECK:     %5 = "cir.struct_element_addr"(%2) <{member_index = 1 : index, member_name = "n"}> : (!cir.ptr<!ty_22struct2EMandalore22>) -> !cir.ptr<!cir.ptr<!void>>
// CHECK:     %6 = cir.const(#cir.null : !cir.ptr<!void>) : !cir.ptr<!void>
// CHECK:     cir.store %6, %5 : !cir.ptr<!void>, cir.ptr <!cir.ptr<!void>>
// CHECK:     %7 = "cir.struct_element_addr"(%2) <{member_index = 2 : index, member_name = "d"}> : (!cir.ptr<!ty_22struct2EMandalore22>) -> !cir.ptr<!s32i>
// CHECK:     %8 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:     cir.store %8, %7 : !s32i, cir.ptr <!s32i>
// CHECK:     cir.return
// CHECK:   }

struct A {
  int a;
};

A get_default() { return A{2}; }

struct S {
  S(A a = get_default());
};

void h() { S s; }

// CHECK: cir.func @_Z1hv()
// CHECK:   %0 = cir.alloca !ty_22struct2ES22, cir.ptr <!ty_22struct2ES22>, ["s", init] {alignment = 1 : i64}
// CHECK:   %1 = cir.alloca !ty_22struct2EA22, cir.ptr <!ty_22struct2EA22>, ["agg.tmp0"] {alignment = 4 : i64}
// CHECK:   %2 = cir.call @_Z11get_defaultv() : () -> !ty_22struct2EA22
// CHECK:   cir.store %2, %1 : !ty_22struct2EA22, cir.ptr <!ty_22struct2EA22>
// CHECK:   %3 = cir.load %1 : cir.ptr <!ty_22struct2EA22>, !ty_22struct2EA22
// CHECK:   cir.call @_ZN1SC1E1A(%0, %3) : (!cir.ptr<!ty_22struct2ES22>, !ty_22struct2EA22) -> ()
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

// CHECK: cir.func linkonce_odr @_ZN5EntryC2Ev(%arg0: !cir.ptr<!ty_22struct2EEntry22>

// CHECK: = "cir.struct_element_addr"(%1) <{member_index = 0 : index, member_name = "procAddr"}> : (!cir.ptr<!ty_22struct2EEntry22>) -> !cir.ptr<!cir.ptr<!cir.func<!u32i (!s32i, !cir.ptr<!s8i>, !cir.ptr<!void>)>>>
