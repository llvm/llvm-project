// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o - | FileCheck %s

struct DummyString {
  DummyString(const char *s) {}
};

void t() {
  DummyString s4 = "yolo";
}

//      CHECK: cir.func linkonce_odr @_ZN11DummyStringC2EPKc
// CHECK-NEXT:     %0 = cir.alloca !cir.ptr<!ty_DummyString>, !cir.ptr<!cir.ptr<!ty_DummyString>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:     %1 = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["s", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %0 : !cir.ptr<!ty_DummyString>, !cir.ptr<!cir.ptr<!ty_DummyString>>
// CHECK-NEXT:     cir.store %arg1, %1 : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CHECK-NEXT:     %2 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_DummyString>>, !cir.ptr<!ty_DummyString>
// CHECK-NEXT:     cir.return

// CHECK-NOT: cir.fun @_ZN11DummyStringC1EPKc

//      CHECK:   cir.func @_Z1tv
// CHECK-NEXT:     %0 = cir.alloca !ty_DummyString, !cir.ptr<!ty_DummyString>, ["s4", init] {alignment = 1 : i64}
// CHECK-NEXT:     %1 = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 5>>
// CHECK-NEXT:     %2 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s8i x 5>>), !cir.ptr<!s8i>
// CHECK-NEXT:     cir.call @_ZN11DummyStringC2EPKc(%0, %2) : (!cir.ptr<!ty_DummyString>, !cir.ptr<!s8i>) -> ()
// CHECK-NEXT:     cir.return

struct B {
  B();
};
B::B() {
}

// CHECK: cir.func @_ZN1BC2Ev(%arg0: !cir.ptr<!ty_B>
// CHECK:   %0 = cir.alloca !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>, ["this", init] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>
// CHECK:   %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_B>>, !cir.ptr<!ty_B>
// CHECK:   cir.return
// CHECK: }
// CHECK: cir.func @_ZN1BC1Ev(!cir.ptr<!ty_B>) alias(@_ZN1BC2Ev)