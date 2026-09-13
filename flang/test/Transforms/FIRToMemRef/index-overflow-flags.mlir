// Test that overflow flags on subscript arithmetic survive the index
// canonicalization performed while lowering fir.array_coor.

// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// An `arith.addi` whose operands need no canonicalization must be reused
// as-is, keeping `overflow<nsw>`.
//
// CHECK-LABEL: func.func @addi_nsw_preserved
// CHECK:       [[I:%.+]]    = memref.load
// CHECK:       [[J:%.+]]    = memref.load
// CHECK:       [[SUM:%.+]]  = arith.addi [[I]], [[J]] overflow<nsw> : i32
// CHECK-NOT:   arith.addi {{.*}} : i32
// CHECK:       [[CAST:%.+]] = arith.index_cast [[SUM]] : i32 to index
func.func @addi_nsw_preserved(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %c100 = arith.constant 100 : index
  %dscope = fir.undefined !fir.dscope
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %a = fir.declare %arg0(%shape) dummy_scope %dscope {uniq_name = "a"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<100xf32>>
  %i = fir.declare %arg1 dummy_scope %dscope {uniq_name = "i"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %j = fir.declare %arg2 dummy_scope %dscope {uniq_name = "j"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %iv = fir.load %i : !fir.ref<i32>
  %jv = fir.load %j : !fir.ref<i32>
  %sum = arith.addi %iv, %jv overflow<nsw> : i32
  %addr = fir.array_coor %a(%shape) %sum : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i32) -> !fir.ref<f32>
  %v = fir.load %addr : !fir.ref<f32>
  return
}

// When canonicalization peels the `arith.extsi` off both operands the add is
// rebuilt at the narrower width, so `overflow<nsw>` must not carry over: a
// 64-bit add that cannot wrap can still wrap in 32 bits.
//
// CHECK-LABEL: func.func @addi_nsw_dropped_on_narrowing
// CHECK:       [[I:%.+]]    = memref.load
// CHECK:       [[J:%.+]]    = memref.load
// CHECK:       [[SUM:%.+]]  = arith.addi [[I]], [[J]] : i32
// CHECK:       [[CAST:%.+]] = arith.index_cast [[SUM]] : i32 to index
func.func @addi_nsw_dropped_on_narrowing(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %c100 = arith.constant 100 : index
  %dscope = fir.undefined !fir.dscope
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %a = fir.declare %arg0(%shape) dummy_scope %dscope {uniq_name = "a"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<100xf32>>
  %i = fir.declare %arg1 dummy_scope %dscope {uniq_name = "i"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %j = fir.declare %arg2 dummy_scope %dscope {uniq_name = "j"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %iv = fir.load %i : !fir.ref<i32>
  %jv = fir.load %j : !fir.ref<i32>
  %ie = arith.extsi %iv : i32 to i64
  %je = arith.extsi %jv : i32 to i64
  %sum = arith.addi %ie, %je overflow<nsw> : i64
  %addr = fir.array_coor %a(%shape) %sum : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
  %v = fir.load %addr : !fir.ref<f32>
  return
}
