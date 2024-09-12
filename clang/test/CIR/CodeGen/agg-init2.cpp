// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: !ty_Zero = !cir.struct<struct "Zero" {!cir.int<u, 8>}>

struct Zero {
  void yolo();
};

void f() {
  Zero z0 = Zero();
  // {} no element init.
  Zero z1 = Zero{};
}

// CHECK: cir.func @_Z1fv()
// CHECK:     %0 = cir.alloca !ty_Zero, !cir.ptr<!ty_Zero>, ["z0", init]
// CHECK:     %1 = cir.alloca !ty_Zero, !cir.ptr<!ty_Zero>, ["z1"]
// CHECK:     cir.call @_ZN4ZeroC1Ev(%0) : (!cir.ptr<!ty_Zero>) -> ()
// CHECK:     cir.return
