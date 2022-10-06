// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

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

//      CHECK: !_22struct2EBar22 = !cir.struct<"struct.Bar", i32, i8>
// CHECK-NEXT: !_22struct2EFoo22 = !cir.struct<"struct.Foo", i32, i8, !cir.struct<"struct.Bar", i32, i8>>

//      CHECK: cir.func linkonce_odr @_ZN3Bar6methodEv(%arg0: !cir.ptr<!_22struct2EBar22>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!_22struct2EBar22>, cir.ptr <!cir.ptr<!_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!_22struct2EBar22>, cir.ptr <!cir.ptr<!_22struct2EBar22>>
// CHECK-NEXT:   %1 = cir.load %0 : cir.ptr <!cir.ptr<!_22struct2EBar22>>, !cir.ptr<!_22struct2EBar22>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method2Ei(%arg0: !cir.ptr<!_22struct2EBar22> {{.*}}, %arg1: i32
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!_22struct2EBar22>, cir.ptr <!cir.ptr<!_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!_22struct2EBar22>, cir.ptr <!cir.ptr<!_22struct2EBar22>>
// CHECK-NEXT:   cir.store %arg1, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!_22struct2EBar22>>, !cir.ptr<!_22struct2EBar22>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN3Bar7method3Ei(%arg0: !cir.ptr<!_22struct2EBar22> {{.*}}, %arg1: i32
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!_22struct2EBar22>, cir.ptr <!cir.ptr<!_22struct2EBar22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!_22struct2EBar22>, cir.ptr <!cir.ptr<!_22struct2EBar22>>
// CHECK-NEXT:   cir.store %arg1, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   %3 = cir.load %0 : cir.ptr <!cir.ptr<!_22struct2EBar22>>, !cir.ptr<!_22struct2EBar22>
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   cir.store %4, %2 : i32, cir.ptr <i32>
// CHECK-NEXT:   %5 = cir.load %2 : cir.ptr <i32>, i32
// CHECK-NEXT:   cir.return %5
// CHECK-NEXT: }

//      CHECK: cir.func @_Z3bazv()
// CHECK-NEXT:   %0 = cir.alloca !_22struct2EBar22, cir.ptr <!_22struct2EBar22>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["result", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !_22struct2EFoo22, cir.ptr <!_22struct2EFoo22>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.call @_ZN3Bar6methodEv(%0) : (!cir.ptr<!_22struct2EBar22>) -> ()
// CHECK-NEXT:   %3 = cir.cst(4 : i32) : i32
// CHECK-NEXT:   cir.call @_ZN3Bar7method2Ei(%0, %3) : (!cir.ptr<!_22struct2EBar22>, i32) -> ()
// CHECK-NEXT:   %4 = cir.cst(4 : i32) : i32
// CHECK-NEXT:   %5 = cir.call @_ZN3Bar7method3Ei(%0, %4) : (!cir.ptr<!_22struct2EBar22>, i32) -> i32
// CHECK-NEXT:   cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
