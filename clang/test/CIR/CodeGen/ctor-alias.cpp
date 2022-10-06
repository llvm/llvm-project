// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o - | FileCheck %s

struct DummyString {
  DummyString(const char *s) {}
};

void t() {
  DummyString s4 = "yolo";
}

//      CHECK: cir.func linkonce_odr @_ZN11DummyStringC2EPKc
// CHECK-NEXT:     %0 = cir.alloca !cir.ptr<!_22struct2EDummyString22>, cir.ptr <!cir.ptr<!_22struct2EDummyString22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:     %1 = cir.alloca !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>, ["s", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %0 : !cir.ptr<!_22struct2EDummyString22>, cir.ptr <!cir.ptr<!_22struct2EDummyString22>>
// CHECK-NEXT:     cir.store %arg1, %1 : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
// CHECK-NEXT:     %2 = cir.load %0 : cir.ptr <!cir.ptr<!_22struct2EDummyString22>>, !cir.ptr<!_22struct2EDummyString22>
// CHECK-NEXT:     cir.return

// CHECK-NOT: cir.fun @_ZN11DummyStringC1EPKc

//      CHECK:   cir.func @_Z1tv
// CHECK-NEXT:     %0 = cir.alloca !_22struct2EDummyString22, cir.ptr <!_22struct2EDummyString22>, ["s4"] {alignment = 1 : i64}
// CHECK-NEXT:     %1 = cir.get_global @".str" : cir.ptr <!cir.array<i8 x 5>>
// CHECK-NEXT:     %2 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<i8 x 5>>), !cir.ptr<i8>
// CHECK-NEXT:     cir.call @_ZN11DummyStringC2EPKc(%0, %2) : (!cir.ptr<!_22struct2EDummyString22>, !cir.ptr<i8>) -> ()
// CHECK-NEXT:     cir.return
