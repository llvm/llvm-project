// RUN: %clang_cc1 -std=c++17 -mconstructor-aliases -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

// TODO: support -mno-constructor-aliases

struct String {
  long size;
  String(const String &s) : size{s.size} {}
// CHECK: cir.func linkonce_odr @_ZN6StringC2ERKS_
// CHECK:     %0 = cir.alloca !cir.ptr<!ty_String>, !cir.ptr<!cir.ptr<!ty_String>>, ["this", init] {alignment = 8 : i64}
// CHECK:     %1 = cir.alloca !cir.ptr<!ty_String>, !cir.ptr<!cir.ptr<!ty_String>>, ["s", init] {alignment = 8 : i64}
// CHECK:     cir.store %arg0, %0
// CHECK:     cir.store %arg1, %1
// CHECK:     %2 = cir.load %0
// CHECK:     %3 = cir.get_member %2[0] {name = "size"}
// CHECK:     %4 = cir.load %1
// CHECK:     %5 = cir.get_member %4[0] {name = "size"}
// CHECK:     %6 = cir.load %5 : !cir.ptr<!s64i>, !s64i
// CHECK:     cir.store %6, %3 : !s64i, !cir.ptr<!s64i>
// CHECK:     cir.return
// CHECK:   }

  String() {}
};

void foo() {
  String s;
  String s1{s};
}
// CHECK: cir.func @_Z3foov() {{.*}} {
// CHECK:  %0 = cir.alloca !ty_String, !cir.ptr<!ty_String>, ["s", init] {alignment = 8 : i64}
// CHECK:  %1 = cir.alloca !ty_String, !cir.ptr<!ty_String>, ["s1", init] {alignment = 8 : i64}
// CHECK:  cir.call @_ZN6StringC2Ev(%0) : (!cir.ptr<!ty_String>) -> ()
// CHECK:  cir.call @_ZN6StringC2ERKS_(%1, %0) : (!cir.ptr<!ty_String>, !cir.ptr<!ty_String>) -> ()
// CHECK:  cir.return
// }
