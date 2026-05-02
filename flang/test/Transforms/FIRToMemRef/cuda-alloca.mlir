// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK:     memref.alloca() {cuf.data_attr = #cuf.cuda<device>} : memref<i32>
// CHECK-NOT: fir.load
// CHECK-NOT: fir.store

func.func @cuf_alloca_(%arg0: !fir.ref<i32> {fir.bindc_name = "r"}) {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.alloca i32 {cuf.data_attr = #cuf.cuda<device>}
  fir.store %c0_i32 to %0 : !fir.ref<i32>
  %1 = fir.load %0 : !fir.ref<i32>
  fir.store %1 to %arg0 : !fir.ref<i32>
  return
}
