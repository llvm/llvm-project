// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

// Checks that we correctly identify when to use the lowering to
// omp.workshare.loop_wrapper

// CHECK-LABEL: @should_parallelize_0
// CHECK: omp.workshare.loop_wrapper
func.func @should_parallelize_0(%arg: !fir.ref<!fir.array<42xi32>>, %idx : index) {
  omp.workshare {
    %c42 = arith.constant 42 : index
    %c1_i32 = arith.constant 1 : i32
    %shape = fir.shape %c42 : (index) -> !fir.shape<1>
    %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
    %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
    ^bb0(%i: index):
      hlfir.yield_element %c1_i32 : i32
    }
    hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
    hlfir.destroy %elemental : !hlfir.expr<42xi32>
    omp.terminator
  }
  return
}

// CHECK-LABEL: @should_parallelize_1
// CHECK: omp.workshare.loop_wrapper
func.func @should_parallelize_1(%arg: !fir.ref<!fir.array<42xi32>>, %idx : index) {
  omp.parallel {
    omp.workshare {
      %c42 = arith.constant 42 : index
      %c1_i32 = arith.constant 1 : i32
      %shape = fir.shape %c42 : (index) -> !fir.shape<1>
      %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
      %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
      ^bb0(%i: index):
        hlfir.yield_element %c1_i32 : i32
      }
      hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
      hlfir.destroy %elemental : !hlfir.expr<42xi32>
      omp.terminator
    }
    omp.terminator
  }
  return
}


// CHECK-LABEL: @should_not_parallelize_0
// CHECK-NOT: omp.workshare.loop_wrapper
func.func @should_not_parallelize_0(%arg: !fir.ref<!fir.array<42xi32>>, %idx : index) {
  omp.workshare {
    omp.single {
      %c42 = arith.constant 42 : index
      %c1_i32 = arith.constant 1 : i32
      %shape = fir.shape %c42 : (index) -> !fir.shape<1>
      %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
      %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
      ^bb0(%i: index):
        hlfir.yield_element %c1_i32 : i32
      }
      hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
      hlfir.destroy %elemental : !hlfir.expr<42xi32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @should_not_parallelize_1
// CHECK-NOT: omp.workshare.loop_wrapper
func.func @should_not_parallelize_1(%arg: !fir.ref<!fir.array<42xi32>>, %idx : index) {
  omp.workshare {
    omp.critical {
      %c42 = arith.constant 42 : index
      %c1_i32 = arith.constant 1 : i32
      %shape = fir.shape %c42 : (index) -> !fir.shape<1>
      %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
      %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
      ^bb0(%i: index):
        hlfir.yield_element %c1_i32 : i32
      }
      hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
      hlfir.destroy %elemental : !hlfir.expr<42xi32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @should_not_parallelize_2
// CHECK-NOT: omp.workshare.loop_wrapper
func.func @should_not_parallelize_2(%arg: !fir.ref<!fir.array<42xi32>>, %idx : index) {
  omp.workshare {
    omp.parallel {
      %c42 = arith.constant 42 : index
      %c1_i32 = arith.constant 1 : i32
      %shape = fir.shape %c42 : (index) -> !fir.shape<1>
      %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
      %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
      ^bb0(%i: index):
        hlfir.yield_element %c1_i32 : i32
      }
      hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
      hlfir.destroy %elemental : !hlfir.expr<42xi32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @should_not_parallelize_3
// CHECK-NOT: omp.workshare.loop_wrapper
func.func @should_not_parallelize_3(%arg: !fir.ref<!fir.array<42xi32>>, %idx : index) {
  omp.workshare {
    omp.parallel {
      omp.workshare {
        omp.parallel {
          %c42 = arith.constant 42 : index
          %c1_i32 = arith.constant 1 : i32
          %shape = fir.shape %c42 : (index) -> !fir.shape<1>
          %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
          %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
          ^bb0(%i: index):
            hlfir.yield_element %c1_i32 : i32
          }
          hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
          hlfir.destroy %elemental : !hlfir.expr<42xi32>
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @should_not_parallelize_4
// CHECK-NOT: omp.workshare.loop_wrapper
func.func @should_not_parallelize_4(%arg: !fir.ref<!fir.array<42xi32>>, %idx : index) {
  omp.workshare {
  ^bb1:
    %c42 = arith.constant 42 : index
    %c1_i32 = arith.constant 1 : i32
    %shape = fir.shape %c42 : (index) -> !fir.shape<1>
    %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
    %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
    ^bb0(%i: index):
      hlfir.yield_element %c1_i32 : i32
    }
    hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
    hlfir.destroy %elemental : !hlfir.expr<42xi32>
    cf.br ^bb2
  ^bb2:
    omp.terminator
  }
  return
}
