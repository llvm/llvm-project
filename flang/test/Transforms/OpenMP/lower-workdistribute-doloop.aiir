// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// CHECK-LABEL:   func.func @x({{.*}})
// CHECK:           omp.teams {
// CHECK:             omp.parallel {
// CHECK:               omp.distribute {
// CHECK:                 omp.wsloop {
// CHECK:                   omp.loop_nest (%[[VAL_1:.*]]) : index = (%[[ARG0:.*]]) to (%[[ARG1:.*]]) inclusive step (%[[ARG2:.*]]) {
// CHECK:                     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:                     fir.store %[[VAL_0]] to %[[ARG4:.*]] : !fir.ref<index>
// CHECK:                     omp.yield
// CHECK:                   }
// CHECK:                 } {omp.composite}
// CHECK:               } {omp.composite}
// CHECK:               omp.terminator
// CHECK:             } {omp.composite}
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  omp.teams {
    omp.workdistribute { 
      fir.do_loop %iv = %lb to %ub step %step unordered {
        %zero = arith.constant 0 : index
        fir.store %zero to %addr : !fir.ref<index>
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
