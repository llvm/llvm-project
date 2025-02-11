// RUN: fir-opt --omp-generic-loop-conversion %s | FileCheck %s

omp.private {type = private} @_QFteams_loopEi_private_i32 : i32

func.func @_QPteams_loop() {
  %i = fir.alloca i32
  omp.teams {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    omp.loop private(@_QFteams_loopEi_private_i32 %i -> %arg2 : !fir.ref<i32>) {
      omp.loop_nest (%arg3) : i32 = (%c0) to (%c10) inclusive step (%c1) {
        fir.store %arg3 to %arg2 : !fir.ref<i32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @_QPteams_loop
// CHECK:         %[[I:.*]] = fir.alloca i32
// CHECK:         omp.teams {
// 
// TODO we probably need to move the `loop_nest` bounds ops from the `teams`
// region to the `parallel` region to avoid making these values `shared`. We can
// find the backward slices of these bounds that are within the `teams` region
// and move these slices to the `parallel` op.

// CHECK:           %[[LB:.*]] = arith.constant 0 : i32
// CHECK:           %[[UB:.*]] = arith.constant 10 : i32
// CHECK:           %[[STEP:.*]] = arith.constant 1 : i32
// 
// CHECK:           omp.parallel private(@{{.*}} %[[I]]
// CHECK-SAME:        -> %[[I_PRIV_ARG:[^[:space:]]+]] : !fir.ref<i32>) {
// CHECK:             omp.distribute {
// CHECK:               omp.wsloop {
// 
// CHECK:                 omp.loop_nest (%{{.*}}) : i32 =
// CHECK-SAME:              (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
// CHECK:                   fir.store %{{.*}} to %[[I_PRIV_ARG]] : !fir.ref<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
