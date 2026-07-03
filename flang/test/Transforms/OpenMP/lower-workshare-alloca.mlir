// RUN: fir-opt --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// Checks that fir.alloca is hoisted out and copyprivate'd
func.func @wsfunc() {
  omp.workshare {
    %c1 = arith.constant 1 : index
    %c42 = arith.constant 42 : index
    %c1_i32 = arith.constant 1 : i32
    %alloc = fir.alloca i32
    fir.store %c1_i32 to %alloc : !fir.ref<i32>
    omp.workshare.loop_wrapper {
      omp.loop_nest (%arg1) : index = (%c1) to (%c42) inclusive step (%c1) {
        "test.test1"(%alloc) : (!fir.ref<i32>) -> ()
        omp.yield
      }
    }
    "test.test2"(%alloc) : (!fir.ref<i32>) -> ()
    omp.terminator
  }
  return
}

// CHECK-LABEL:   func.func private @_workshare_copy_i32(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.ref<i32>)
// CHECK-SAME:                   attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:           fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @wsfunc() {
// CHECK:           %[[VAL_0:.*]] = fir.alloca i32
// CHECK:           omp.single copyprivate(%[[VAL_0]] -> @_workshare_copy_i32 : !fir.ref<i32>) {
// CHECK:             %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:             fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<i32>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 42 : index
// CHECK:           omp.wsloop {
// CHECK:             omp.loop_nest (%[[VAL_4:.*]]) : index = (%[[VAL_2]]) to (%[[VAL_3]]) inclusive step (%[[VAL_2]]) {
// CHECK:               "test.test1"(%[[VAL_0]]) : (!fir.ref<i32>) -> ()
// CHECK:               omp.yield
// CHECK:             }
// CHECK:           }
// CHECK:           omp.single nowait {
// CHECK:             "test.test2"(%[[VAL_0]]) : (!fir.ref<i32>) -> ()
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           omp.barrier
// CHECK:           return
// CHECK:         }

