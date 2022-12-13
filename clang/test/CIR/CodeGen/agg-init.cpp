// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Zero {
  void yolo(); 
};

void f() {
  Zero z0 = Zero();
  // {} no element init.
  Zero z1 = Zero{};
}

// CHECK: cir.func @_Z1fv() {
// CHECK:     %0 = cir.alloca !ty_22struct2EZero22, cir.ptr <!ty_22struct2EZero22>, ["z0", init]
// CHECK:     %1 = cir.alloca !ty_22struct2EZero22, cir.ptr <!ty_22struct2EZero22>, ["z1"]
// CHECK:     cir.call @_ZN4ZeroC1Ev(%0) : (!cir.ptr<!ty_22struct2EZero22>) -> ()
// CHECK:     cir.return
