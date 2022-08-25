// RUN: fir-opt --split-input-file --simplify-intrinsics='enable-experimental=true' %s | FileCheck %s

// check that a simple function compiled with experimental enabled
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @experimental_1(%arg0: !fir.ref<i32> {fir.bindc_name = "a"}) -> i32 {
    %c10 = arith.constant 10 : i32
    return %c10 : i32
  }
}

// CHECK-LABEL:   func.func @experimental_1(
// CHECK:           return %{{.*}} : i32
// CHECK:         }
