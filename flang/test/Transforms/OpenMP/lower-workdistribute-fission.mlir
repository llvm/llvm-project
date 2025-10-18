// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// CHECK-LABEL:   func.func @test_fission_workdistribute(
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:           fir.store %[[VAL_3]] to %[[ARG2:.*]] : !fir.ref<f32>
// CHECK:           omp.teams {
// CHECK:             omp.parallel {
// CHECK:               omp.distribute {
// CHECK:                 omp.wsloop {
// CHECK:                   omp.loop_nest (%[[VAL_4:.*]]) : index = (%[[VAL_0]]) to (%[[VAL_2]]) inclusive step (%[[VAL_1]]) {
// CHECK:                     %[[VAL_5:.*]] = fir.coordinate_of %[[ARG0:.*]], %[[VAL_4]] : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
// CHECK:                     %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<f32>
// CHECK:                     %[[VAL_7:.*]] = fir.coordinate_of %[[ARG1:.*]], %[[VAL_4]] : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
// CHECK:                     fir.store %[[VAL_6]] to %[[VAL_7]] : !fir.ref<f32>
// CHECK:                     omp.yield
// CHECK:                   }
// CHECK:                 } {omp.composite}
// CHECK:               } {omp.composite}
// CHECK:               omp.terminator
// CHECK:             } {omp.composite}
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           fir.call @regular_side_effect_func(%[[ARG2:.*]]) : (!fir.ref<f32>) -> ()
// CHECK:           fir.call @my_fir_parallel_runtime_func(%[[ARG3:.*]]) : (!fir.ref<f32>) -> ()
// CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_1]] {
// CHECK:             %[[VAL_9:.*]] = fir.coordinate_of %[[ARG0]], %[[VAL_8]] : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
// CHECK:             fir.store %[[VAL_3]] to %[[VAL_9]] : !fir.ref<f32>
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = fir.load %[[ARG2:.*]] : !fir.ref<f32>
// CHECK:           fir.store %[[VAL_10]] to %[[ARG3:.*]] : !fir.ref<f32>
// CHECK:           return
// CHECK:         }
module {
func.func @regular_side_effect_func(%arg0: !fir.ref<f32>) {
  return
}
func.func @my_fir_parallel_runtime_func(%arg0: !fir.ref<f32>) attributes {fir.runtime} {
  return
}
func.func @test_fission_workdistribute(%arr1: !fir.ref<!fir.array<10xf32>>, %arr2: !fir.ref<!fir.array<10xf32>>, %scalar_ref1: !fir.ref<f32>, %scalar_ref2: !fir.ref<f32>) {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c9_idx = arith.constant 9 : index
  %float_val = arith.constant 5.0 : f32
  omp.teams   {
    omp.workdistribute   {
      fir.store %float_val to %scalar_ref1 : !fir.ref<f32>
      fir.do_loop %iv = %c0_idx to %c9_idx step %c1_idx unordered {
        %elem_ptr_arr1 = fir.coordinate_of %arr1, %iv : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %loaded_val_loop1 = fir.load %elem_ptr_arr1 : !fir.ref<f32>
        %elem_ptr_arr2 = fir.coordinate_of %arr2, %iv : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        fir.store %loaded_val_loop1 to %elem_ptr_arr2 : !fir.ref<f32>
      }
      fir.call @regular_side_effect_func(%scalar_ref1) : (!fir.ref<f32>) -> ()
      fir.call @my_fir_parallel_runtime_func(%scalar_ref2) : (!fir.ref<f32>) -> ()
      fir.do_loop %jv = %c0_idx to %c9_idx step %c1_idx {
        %elem_ptr_ordered_loop = fir.coordinate_of %arr1, %jv : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        fir.store %float_val to %elem_ptr_ordered_loop : !fir.ref<f32>
      }
      %loaded_for_hoist = fir.load %scalar_ref1 : !fir.ref<f32>
      fir.store %loaded_for_hoist to %scalar_ref2 : !fir.ref<f32>
      omp.terminator  
    }
    omp.terminator
  }
  return
}
}
