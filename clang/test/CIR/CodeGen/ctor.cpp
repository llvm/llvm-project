// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Struk {
  int a;
  Struk() {}
};

void baz() {
  Struk s;
}

// CHECK: !rec_Struk = !cir.record<struct "Struk" {!s32i}>

// Note: In the absence of the '-mconstructor-aliases' option, we emit two
//       constructors here. The handling of constructor aliases is currently
//       NYI, but when it is added this test should be updated to add a RUN
//       line that passes '-mconstructor-aliases' to clang_cc1.
// CHECK:   cir.func @_ZN5StrukC2Ev(%arg0: !cir.ptr<!rec_Struk>
// CHECK-NEXT:     %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>
// CHECK-NEXT:     %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_Struk>>, !cir.ptr<!rec_Struk>
// CHECK-NEXT:     cir.return

// CHECK:   cir.func @_ZN5StrukC1Ev(%arg0: !cir.ptr<!rec_Struk>
// CHECK-NEXT:     %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>
// CHECK-NEXT:     %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_Struk>>, !cir.ptr<!rec_Struk>
// CHECK-NEXT:     cir.call @_ZN5StrukC2Ev(%[[THIS]]) : (!cir.ptr<!rec_Struk>) -> ()
// CHECK-NEXT:     cir.return

// CHECK:   cir.func @_Z3bazv()
// CHECK-NEXT:     %[[S_ADDR:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["s", init] {alignment = 4 : i64}
// CHECK-NEXT:     cir.call @_ZN5StrukC1Ev(%[[S_ADDR]]) : (!cir.ptr<!rec_Struk>) -> ()
// CHECK-NEXT:     cir.return
