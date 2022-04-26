// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Struk {
  int a;
  Struk() {}
  void test() {}
};

void baz() {
  Struk s;
}

//      CHECK: !_22struct2EStruk22 = !cir.struct<"struct.Struk", i32>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func @_Z3bazv()
// CHECK-NEXT:     %0 = cir.alloca !_22struct2EStruk22, cir.ptr <!_22struct2EStruk22>, ["s", uninitialized] {alignment = 4 : i64}
// CHECK-NEXT:     call @_ZN5StrukC1Ev(%0) : (!cir.ptr<!_22struct2EStruk22>) -> ()
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @_ZN5StrukC1Ev(%arg0: !cir.ptr<!_22struct2EStruk22>
// CHECK-NEXT:     %0 = cir.alloca !cir.ptr<!_22struct2EStruk22>, cir.ptr <!cir.ptr<!_22struct2EStruk22>>, ["this", paraminit] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %0 : !cir.ptr<!_22struct2EStruk22>, cir.ptr <!cir.ptr<!_22struct2EStruk22>>
// CHECK-NEXT:     %1 = cir.load %0 : cir.ptr <!cir.ptr<!_22struct2EStruk22>>, !cir.ptr<!_22struct2EStruk22>
// CHECK-NEXT:     call @_ZN5StrukC2Ev(%1) : (!cir.ptr<!_22struct2EStruk22>) -> ()
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @_ZN5StrukC2Ev(%arg0: !cir.ptr<!_22struct2EStruk22>
// CHECK-NEXT:     %0 = cir.alloca !cir.ptr<!_22struct2EStruk22>, cir.ptr <!cir.ptr<!_22struct2EStruk22>>, ["this", paraminit] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %0 : !cir.ptr<!_22struct2EStruk22>, cir.ptr <!cir.ptr<!_22struct2EStruk22>>
// CHECK-NEXT:     %1 = cir.load %0 : cir.ptr <!cir.ptr<!_22struct2EStruk22>>, !cir.ptr<!_22struct2EStruk22>
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
