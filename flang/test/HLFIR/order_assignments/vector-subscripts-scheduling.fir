// Test local alloca and store inside hlfir.region_assign do not trigger the
// creation of a temporary for the LHS.

// RUN: fir-opt -o - -lower-hlfir-ordered-assignments --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s
// REQUIRES: asserts

func.func @simple(%arg0: !fir.ref<!fir.array<100x10xf32>> , %arg1: !fir.ref<!fir.array<10xi64>> , %arg2: !fir.ref<!fir.array<10xf32>>, %i: i64, %f: f32) {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100, %c10 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFsimpleEx"} : (!fir.ref<!fir.array<100x10xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x10xf32>>, !fir.ref<!fir.array<100x10xf32>>)
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
  hlfir.region_assign {
    hlfir.yield %f : f32
  } to {
    %local_temp = fir.alloca i64
    fir.store %i to %local_temp : !fir.ref<i64>
    %icopy = fir.load %local_temp : !fir.ref<i64>
    hlfir.elemental_addr %2 : !fir.shape<1> {
    ^bb0(%arg3: index):
      %5 = hlfir.designate %3#0 (%arg3)  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
      %6 = fir.load %5 : !fir.ref<i64>
      %7 = hlfir.designate %1#0 (%icopy, %6)  : (!fir.ref<!fir.array<100x10xf32>>, i64, i64) -> !fir.ref<f32>
      hlfir.yield %7 : !fir.ref<f32>
    }
  }
  return
}

// CHECK: run 1 evaluate: region_assign
