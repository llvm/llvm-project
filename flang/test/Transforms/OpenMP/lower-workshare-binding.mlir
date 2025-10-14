// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// Checks that the omp.workshare.loop_wrapper binds to the correct omp.workshare

func.func @wsfunc() {
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  omp.parallel {
    omp.workshare nowait {
      omp.parallel {
        omp.workshare nowait {
          omp.workshare.loop_wrapper {
            omp.loop_nest (%arg1) : index = (%c1) to (%c42) inclusive step (%c1) {
              "test.test2"() : () -> ()
              omp.yield
            }
          }
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

// CHECK-LABEL:   func.func @wsfunc() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           omp.parallel {
// CHECK:             omp.single nowait {
// CHECK:               omp.parallel {
// CHECK:                 omp.wsloop nowait {
// CHECK:                   omp.loop_nest (%[[VAL_2:.*]]) : index = (%[[VAL_0]]) to (%[[VAL_1]]) inclusive step (%[[VAL_0]]) {
// CHECK:                     "test.test2"() : () -> ()
// CHECK:                     omp.yield
// CHECK:                   }
// CHECK:                 }
// CHECK:                 omp.terminator
// CHECK:               }
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

