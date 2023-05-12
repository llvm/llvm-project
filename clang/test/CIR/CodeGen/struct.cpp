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

//      CHECK: !ty_22struct2EBar22 = !cir.struct<"struct.Bar", i32, i8>
//      CHECK: !ty_22struct2EMandalore22 = !cir.struct<"struct.Mandalore", i32, !cir.ptr<i8>, i32, #cir.recdecl.ast>
//      CHECK: !ty_22struct2Eincomplete22 = !cir.struct<"struct.incomplete", incomplete
//      CHECK: !ty_22class2EAdv22 = !cir.struct<"class.Adv", !ty_22struct2EMandalore22>
//      CHECK: !ty_22struct2EFoo22 = !cir.struct<"struct.Foo", i32, i8, !ty_22struct2EBar22>

//      CHECK: cir.func linkonce_odr @_ZN3Bar6methodEv(%arg0: !cir.ptr<!ty_22struct2EBar22>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>
// CHECK-NEXT:   %1 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, !cir.ptr<!ty_22struct2EBar22>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method2Ei(%arg0: !cir.ptr<!ty_22struct2EBar22> {{.*}}, %arg1: i32
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>
// CHECK-NEXT:   cir.store %arg1, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, !cir.ptr<!ty_22struct2EBar22>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method3Ei(%arg0: !cir.ptr<!ty_22struct2EBar22> {{.*}}, %arg1: i32
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22struct2EBar22>, cir.ptr <!cir.ptr<!ty_22struct2EBar22>>
// CHECK-NEXT:   cir.store %arg1, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   %3 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22struct2EBar22>>, !cir.ptr<!ty_22struct2EBar22>
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   cir.store %4, %2 : i32, cir.ptr <i32>
// CHECK-NEXT:   %5 = cir.load %2 : cir.ptr <i32>, i32
// CHECK-NEXT:   cir.return %5
// CHECK-NEXT: }

//      CHECK: cir.func @_Z3bazv()
// CHECK-NEXT:   %0 = cir.alloca !ty_22struct2EBar22, cir.ptr <!ty_22struct2EBar22>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["result", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.call @_ZN3Bar6methodEv(%0) : (!cir.ptr<!ty_22struct2EBar22>) -> ()
// CHECK-NEXT:   %3 = cir.const(4 : i32) : i32
// CHECK-NEXT:   cir.call @_ZN3Bar7method2Ei(%0, %3) : (!cir.ptr<!ty_22struct2EBar22>, i32) -> ()
// CHECK-NEXT:   %4 = cir.const(4 : i32) : i32
// CHECK-NEXT:   %5 = cir.call @_ZN3Bar7method3Ei(%0, %4) : (!cir.ptr<!ty_22struct2EBar22>, i32) -> i32
// CHECK-NEXT:   cir.store %5, %1 : i32, cir.ptr <i32>
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
// CHECK:     %2 = "cir.struct_element_addr"(%1) <{member_name = "x"}> : (!cir.ptr<!ty_22class2EAdv22>) -> !cir.ptr<!ty_22struct2EMandalore22>
// CHECK:     %3 = "cir.struct_element_addr"(%2) <{member_name = "w"}> : (!cir.ptr<!ty_22struct2EMandalore22>) -> !cir.ptr<i32>
// CHECK:     %4 = cir.const(1000024001 : i32) : i32
// CHECK:     cir.store %4, %3 : i32, cir.ptr <i32>
// CHECK:     %5 = "cir.struct_element_addr"(%2) <{member_name = "n"}> : (!cir.ptr<!ty_22struct2EMandalore22>) -> !cir.ptr<!cir.ptr<i8>>
// CHECK:     %6 = cir.const(#cir.null : !cir.ptr<i8>) : !cir.ptr<i8>
// CHECK:     cir.store %6, %5 : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
// CHECK:     %7 = "cir.struct_element_addr"(%2) <{member_name = "d"}> : (!cir.ptr<!ty_22struct2EMandalore22>) -> !cir.ptr<i32>
// CHECK:     %8 = cir.const(0 : i32) : i32
// CHECK:     cir.store %8, %7 : i32, cir.ptr <i32>
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

// CHECK: cir.func @_Z1hv() {
// CHECK:   %0 = cir.alloca !ty_22struct2ES22, cir.ptr <!ty_22struct2ES22>, ["s", init] {alignment = 1 : i64}
// CHECK:   %1 = cir.alloca !ty_22struct2EA22, cir.ptr <!ty_22struct2EA22>, ["agg.tmp0"] {alignment = 4 : i64}
// CHECK:   %2 = cir.call @_Z11get_defaultv() : () -> !ty_22struct2EA22
// CHECK:   cir.store %2, %1 : !ty_22struct2EA22, cir.ptr <!ty_22struct2EA22>
// CHECK:   %3 = cir.load %1 : cir.ptr <!ty_22struct2EA22>, !ty_22struct2EA22
// CHECK:   cir.call @_ZN1SC1E1A(%0, %3) : (!cir.ptr<!ty_22struct2ES22>, !ty_22struct2EA22) -> ()
// CHECK:   cir.return
// CHECK: }
