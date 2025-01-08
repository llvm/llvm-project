// RUN: fir-opt --omp-generic-loop-conversion %s | FileCheck %s

omp.private {type = private} @_QFtarget_teams_loopEi_private_ref_i32 : !fir.ref<i32> alloc {
^bb0(%arg0: !fir.ref<i32>):
  omp.yield(%arg0 : !fir.ref<i32>)
}

func.func @_QPtarget_teams_loop() {
  %i = fir.alloca i32
  %i_map = omp.map.info var_ptr(%i : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
  omp.target map_entries(%i_map -> %arg0 : !fir.ref<i32>) {
    omp.teams {
      %c0 = arith.constant 0 : i32
      %c10 = arith.constant 10 : i32
      %c1 = arith.constant 1 : i32
      omp.loop private(@_QFtarget_teams_loopEi_private_ref_i32 %arg0 -> %arg2 : !fir.ref<i32>) {
        omp.loop_nest (%arg3) : i32 = (%c0) to (%c10) inclusive step (%c1) {
          fir.store %arg3 to %arg2 : !fir.ref<i32>
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @_QPtarget_teams_loop
// CHECK:         omp.target map_entries(
// CHECK-SAME:      %{{.*}} -> %[[I_ARG:[^[:space:]]+]] : {{.*}}) {
// 
// CHECK:           omp.teams {
// 
// TODO we probably need to move the `loop_nest` bounds ops from the `teams`
// region to the `parallel` region to avoid making these values `shared`. We can
// find the backward slices of these bounds that are within the `teams` region
// and move these slices to the `parallel` op.

// CHECK:             %[[LB:.*]] = arith.constant 0 : i32
// CHECK:             %[[UB:.*]] = arith.constant 10 : i32
// CHECK:             %[[STEP:.*]] = arith.constant 1 : i32
// 
// CHECK:             omp.parallel private(@{{.*}} %[[I_ARG]]
// CHECK-SAME:          -> %[[I_PRIV_ARG:[^[:space:]]+]] : !fir.ref<i32>) {
// CHECK:               omp.distribute {
// CHECK:                 omp.wsloop {
// 
// CHECK:                   omp.loop_nest (%{{.*}}) : i32 = 
// CHECK-SAME:                (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
// CHECK:                     fir.store %{{.*}} to %[[I_PRIV_ARG]] : !fir.ref<i32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
