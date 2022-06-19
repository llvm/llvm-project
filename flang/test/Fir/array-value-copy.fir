// Test for the array-value-copy-pass
// RUN: fir-opt --split-input-file --array-value-copy %s | FileCheck %s

// Test simple fir.array_load/fir.array_fetch conversion to fir.array_coor
func.func @array_fetch_conversion(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m: index, %n: index) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %s = fir.shape %m, %n : (index, index) -> !fir.shape<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  %f = fir.array_fetch %av1, %c10, %c20 : (!fir.array<?x?xf32>, index, index) -> f32
  return
}

// CHECK-LABEL: func @array_fetch_conversion(
// CHECK-SAME:                               %[[ARRAY:.*]]: !fir.ref<!fir.array<?x?xf32>>,
// CHECK-SAME:                               %[[ARG1:.*]]: index,
// CHECK-SAME:                               %[[ARG2:.*]]: index) {
// CHECK:         %{{.*}} = fir.shape %[[ARG1]], %[[ARG2]] : (index, index) -> !fir.shape<2>
// CHECK:         %{{.*}} = fir.undefined !fir.array<?x?xf32>
// CHECK:         %[[VAL_0:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:         %[[VAL_1:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NOT:     fir.array_load
// CHECK-NOT:     fir.array_fetch
// CHECK:         %{{.*}} = fir.array_coor %arg0(%0) %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
// CHECK:         %{{.*}} = fir.load %4 : !fir.ref<f32>

// -----

// Test simple fir.array_load/fir.array_update conversion without copy-in/copy-out
func.func @array_update_conversion(%arr1 : !fir.box<!fir.array<?x?xf32>>, %m: index, %n: index) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 2.0 : f32
  %s = fir.shape %m, %n : (index, index) -> !fir.shape<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  %av2 = fir.array_update %av1, %f, %c1, %c1 : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  return
}

// CHECK-LABEL: func @array_update_conversion
// CHECK-NOT:     fir.array_load
// CHECK-NOT:     fir.array_update
// CHECK:         %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:         %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:         %[[ARRAY_COOR:.*]] = fir.array_coor{{.*}}-> !fir.ref<f32>
// CHECK:         fir.store %{{.*}} to %[[ARRAY_COOR]] : !fir.ref<f32>

// -----

// Test simple fir.array_load/fir.array_update conversion without copy-in/copy-out
func.func @array_update_conversion(%arr1 : !fir.box<!fir.array<?x?xf32>>, %m: index, %n: index, %cond: i1) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 2.0 : f32
  %g = arith.constant 4.0 : f32
  %s = fir.shape %m, %n : (index, index) -> !fir.shape<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  fir.if %cond {
    %av2 = fir.array_update %av1, %f, %c1, %c1 : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  } else {
    %av2 = fir.array_update %av1, %g, %c1, %c1 : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  }
  return
}

// -----

// Test fir.array_load/fir.array_fetch/fir.array_update conversion with
// an introduced copy-in/copy-out.
// 
// This test corresponds to a simplified FIR version of the following Fortran
// code.
// ```
//   integer :: i(10)
//   i = i(10:1:-1)
//  end
// ```

func.func @conversion_with_temporary(%arr0 : !fir.ref<!fir.array<10xi32>>) {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arr0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
  %c10_i64 = arith.constant 10 : i64
  %3 = fir.convert %c10_i64 : (i64) -> index
  %c1_i64 = arith.constant 1 : i64
  %c-1_i64 = arith.constant -1 : i64
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5 = fir.slice %c10_i64, %c1_i64, %c-1_i64 : (i64, i64, i64) -> !fir.slice<1>
  %6 = fir.array_load %arr0(%4) [%5] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %7 = arith.subi %3, %c1 : index
  %8 = fir.do_loop %arg0 = %c0 to %7 step %c1 unordered iter_args(%arg1 = %2) -> (!fir.array<10xi32>) {
    %9 = fir.array_fetch %6, %arg0 : (!fir.array<10xi32>, index) -> i32
    %10 = fir.array_update %arg1, %9, %arg0 : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
    fir.result %10 : !fir.array<10xi32>
  }
  fir.array_merge_store %2, %8 to %arr0 : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
  return
}

// CHECK-LABEL: func @conversion_with_temporary(
// CHECK-SAME:                                  %[[ARR0:.*]]: !fir.ref<!fir.array<10xi32>>)
// Allocation of temporary array.
// CHECK:         %[[TEMP:.*]] = fir.allocmem !fir.array<10xi32>
// Copy of original array to temp.
// CHECK:         fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:           %[[COOR1:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:           %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<i32>
// CHECK:           fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:         }
// Perform the assignment i = i(10:1:-1) using the temporary array.
// CHECK:         %{{.*}} = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<10xi32>) {
// CHECK-NOT:       %{{.*}} = fir.array_fetch
// CHECK-NOT:       %{{.*}} = fir.array_update
// CHECK:           %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) [%{{.*}}] %{{.*}} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<i32>
// CHECK:           %[[COOR1:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:           fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:           fir.result %{{.*}} : !fir.array<10xi32>
// CHECK:         }
// Copy the result back to the original array.
// CHECK:         fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           %[[COOR0:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:           %[[COOR1:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:           %[[LOAD0:.*]] = fir.load %[[COOR0:.*]] : !fir.ref<i32>
// CHECK:           fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:         }
// Free temporary array.
// CHECK:         fir.freemem %[[TEMP]] : !fir.heap<!fir.array<10xi32>>

// -----

// Test fir.array_load/fir.array_fetch/fir.array_update conversion with
// an introduced copy-in/copy-out on a multidimensional array.

func.func @conversion_with_temporary_multidim(%0: !fir.ref<!fir.array<10x5xi32>>) {
  %c10 = arith.constant 10 : index
  %c5 = arith.constant 5 : index
  %1 = fir.shape %c10, %c5 : (index, index) -> !fir.shape<2>
  %2 = fir.array_load %0(%1) : (!fir.ref<!fir.array<10x5xi32>>, !fir.shape<2>) -> !fir.array<10x5xi32>
  %c10_i64 = arith.constant 10 : i64
  %3 = fir.convert %c10_i64 : (i64) -> index
  %c5_i64 = arith.constant 5 : i64
  %4 = fir.convert %c5_i64 : (i64) -> index
  %c1 = arith.constant 1 : index
  %c10_i64_0 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64
  %c-1_i64 = arith.constant -1 : i64
  %5 = arith.addi %c1, %c5 : index
  %6 = arith.subi %5, %c1 : index
  %c1_i64_1 = arith.constant 1 : i64
  %7 = fir.shape %c10, %c5 : (index, index) -> !fir.shape<2>
  %8 = fir.slice %c10_i64_0, %c1_i64, %c-1_i64, %c1, %6, %c1_i64_1 : (i64, i64, i64, index, index, i64) -> !fir.slice<2>
  %9 = fir.array_load %0(%7) [%8] : (!fir.ref<!fir.array<10x5xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x5xi32>
  %c1_2 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %10 = arith.subi %3, %c1_2 : index
  %11 = arith.subi %4, %c1_2 : index
  %12 = fir.do_loop %arg0 = %c0 to %11 step %c1_2 unordered iter_args(%arg1 = %2) -> (!fir.array<10x5xi32>) {
    %13 = fir.do_loop %arg2 = %c0 to %10 step %c1_2 unordered iter_args(%arg3 = %arg1) -> (!fir.array<10x5xi32>) {
      %14 = fir.array_fetch %9, %arg2, %arg0 : (!fir.array<10x5xi32>, index, index) -> i32
      %15 = fir.array_update %arg3, %14, %arg2, %arg0 : (!fir.array<10x5xi32>, i32, index, index) -> !fir.array<10x5xi32>
      fir.result %15 : !fir.array<10x5xi32>
    }
    fir.result %13 : !fir.array<10x5xi32>
  }
  fir.array_merge_store %2, %12 to %0 : !fir.array<10x5xi32>, !fir.array<10x5xi32>, !fir.ref<!fir.array<10x5xi32>>
  return
}

// CHECK-LABEL: func @conversion_with_temporary_multidim(
// CHECK-SAME:                                           %[[ARR0:.*]]: !fir.ref<!fir.array<10x5xi32>>) {
// CHECK:         %[[CST10:.*]] = arith.constant 10 : index
// CHECK:         %[[CST5:.*]] = arith.constant 5 : index
// CHECK:         %[[TEMP:.*]] = fir.allocmem !fir.array<10x5xi32>
// CHECK:         %[[IDX5:.*]] = fir.convert %[[CST5]] : (index) -> index
// CHECK:         %[[UB5:.*]] = arith.subi %[[IDX5]], %{{.*}} : index
// CHECK:         fir.do_loop %[[INDUC0:.*]] = %{{.*}} to %[[UB5]] step %{{.*}} {
// CHECK:           %[[IDX10:.*]] = fir.convert %[[CST10]] : (index) -> index
// CHECK:           %[[UB10:.*]] = arith.subi %[[IDX10]], %{{.*}} : index
// CHECK:           fir.do_loop %[[INDUC1:.*]] = %{{.*}} to %[[UB10]] step %{{.*}} {
// CHECK:             %[[IDX1:.*]] = arith.addi %[[INDUC1]], %{{.*}} : index
// CHECK:             %[[IDX2:.*]] = arith.addi %[[INDUC0]], %{{.*}} : index
// CHECK:             %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %[[IDX1:.*]], %[[IDX2:.*]] : (!fir.ref<!fir.array<10x5xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
// CHECK:             %[[COOR1:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}}, %{{.*}} : (!fir.heap<!fir.array<10x5xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
// CHECK:             %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<i32>
// CHECK:             fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:         %{{.*}} = fir.do_loop %[[INDUC0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<10x5xi32>) {
// CHECK:           %{{.*}} = fir.do_loop %[[INDUC1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<10x5xi32>) {
// CHECK:             %[[IDX1:.*]] = arith.addi %[[INDUC1]], %{{.*}} : index
// CHECK:             %[[IDX2:.*]] = arith.addi %[[INDUC0]], %{{.*}} : index
// CHECK-NOT:         %{{.*}} = fir.array_fetch
// CHECK-NOT:         %{{.*}} = fir.array_update
// CHECK:             %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) [%{{.*}}] %[[IDX1]], %[[IDX2]] : (!fir.ref<!fir.array<10x5xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
// CHECK:             %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<i32>
// CHECK:             %[[COOR1:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}}, %{{.*}} : (!fir.heap<!fir.array<10x5xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:         %[[IDX5:.*]] = fir.convert %[[CST5]] : (index) -> index
// CHECK:         %[[UB5:.*]] = arith.subi %[[IDX5]], %{{.*}} : index
// CHECK:         fir.do_loop %[[INDUC0:.*]] = %{{.*}} to %[[UB5]] step %{{.*}} {
// CHECK:           %[[IDX10:.*]] = fir.convert %[[CST10]] : (index) -> index
// CHECK:           %[[UB10:.*]] = arith.subi %[[IDX10]], %{{.*}} : index
// CHECK:           fir.do_loop %[[INDUC1:.*]] = %{{.*}} to %[[UB10]] step %{{.*}} {
// CHECK:             %[[IDX1:.*]] = arith.addi %[[INDUC1]], %{{.*}} : index
// CHECK:             %[[IDX2:.*]] = arith.addi %[[INDUC0]], %{{.*}} : index  
// CHECK:             %[[COOR0:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %[[IDX1]], %[[IDX2]] : (!fir.heap<!fir.array<10x5xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
// CHECK:             %[[COOR1:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}}, %{{.*}} : (!fir.ref<!fir.array<10x5xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
// CHECK:             %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<i32>
// CHECK:             fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:         fir.freemem %[[TEMP]] : !fir.heap<!fir.array<10x5xi32>>

// -----

// Test fir.array_modify conversion with no overlap.
func.func @array_modify_no_overlap(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %2) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %3, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6:2 = fir.array_modify %arg3, %arg2 : (!fir.array<100xf32>, index) -> (!fir.ref<f32>, !fir.array<100xf32>)
    fir.store %5 to %0 : !fir.ref<f32>
    fir.call @user_defined_assignment(%6#0, %0) : (!fir.ref<f32>, !fir.ref<f32>) -> ()
    fir.result %6#1 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %4 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}

func.func private @user_defined_assignment(!fir.ref<f32>, !fir.ref<f32>)

// CHECK-LABEL:   func @array_modify_no_overlap(
// CHECK-SAME:                                  %[[ARR0:.*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                                  %[[ARR1:.*]]: !fir.ref<!fir.array<100xf32>>) {
// CHECK:           %[[VAR0:.*]] = fir.alloca f32
// CHECK-COUNT-1:   %{{.*}} = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<100xf32>) {
// CHECK-NOT:       %{{.*}} = fir.array_fetch
// CHECK-NOT:       %{{.*}} = fir.array_modify
// CHECK:           %[[COOR0:.*]] = fir.array_coor %arg1(%1) %5 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:           %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<f32>
// CHECK:           %[[COOR1:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:           fir.store %[[LOAD0]] to %[[VAR0]] : !fir.ref<f32>
// CHECK:           fir.call @{{.*}}(%[[COOR1]], %[[VAR0]]) : (!fir.ref<f32>, !fir.ref<f32>) -> ()

// -----

// Test fir.array_modify conversion with an overlap.
// Test user_defined_assignment(arg0(:), arg0(100:1:-1))
func.func @array_modify_overlap(%arg0: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.slice %c100, %c1, %c-1 : (index, index, index) -> !fir.slice<1>
  %4 = fir.array_load %arg0(%1) [%3] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  %5 = fir.do_loop %arg1 = %c0 to %c99 step %c1 unordered iter_args(%arg2 = %2) -> (!fir.array<100xf32>) {
    %6 = fir.array_fetch %4, %arg1 : (!fir.array<100xf32>, index) -> f32
    %7:2 = fir.array_modify %arg2, %arg1 : (!fir.array<100xf32>, index) -> (!fir.ref<f32>, !fir.array<100xf32>)
    fir.store %6 to %0 : !fir.ref<f32>
    fir.call @user_defined_assignment(%7#0, %0) : (!fir.ref<f32>, !fir.ref<f32>) -> ()
    fir.result %7#1 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %5 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}

func.func private @user_defined_assignment(!fir.ref<f32>, !fir.ref<f32>)

// CHECK-LABEL:   func @array_modify_overlap(
// CHECK-SAME:                               %[[ARR0:.*]]: !fir.ref<!fir.array<100xf32>>) {
// CHECK:           %[[VAR0:.*]] = fir.alloca f32
// Allocate the temporary array.
// CHECK:           %[[TEMP:.*]] = fir.allocmem !fir.array<100xf32>
// Copy original array to temp.
// CHECK:           fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[COOR1:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<f32>
// CHECK:             fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<f32>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = fir.undefined !fir.array<100xf32>
// CHECK:           %[[VAL_23:.*]] = fir.undefined !fir.array<100xf32>
// CHECK-NOT:         %{{.*}} = fir.array_fetch
// CHECK-NOT:         %{{.*}} = fir.array_modify
// CHECK:             %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) {{\[}}%{{.*}}] %{{.*}} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
// CHECK:             %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<f32>
// CHECK:             %[[COOR1:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             fir.store %[[LOAD0]] to %[[VAR0]] : !fir.ref<f32>
// CHECK:             fir.call @user_defined_assignment(%[[COOR1]], %[[VAR0]]) : (!fir.ref<f32>, !fir.ref<f32>) -> ()
// CHECK:           }
// Copy back result to original array from temp.
// CHECK:           fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             %[[COOR0:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[COOR1:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<f32>
// CHECK:             fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<f32>
// CHECK:           }
// Free the temporary array.
// CHECK:           fir.freemem %[[TEMP]] : !fir.heap<!fir.array<100xf32>>
// CHECK:           return
// CHECK:         }

// -----

// Test array of types with no overlap
func.func @array_of_types() {
  %0 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QEj"}
  %1 = fir.address_of(@_QEtypes) : !fir.ref<!fir.array<10x!fir.type<_QTd{i:!fir.array<10xi32>}>>>
  %c1_i32 = arith.constant 1 : i32
  %2 = fir.convert %c1_i32 : (i32) -> index
  %c10_i32 = arith.constant 10 : i32
  %3 = fir.convert %c10_i32 : (i32) -> index
  %c1 = arith.constant 1 : index
  %4 = fir.do_loop %arg0 = %2 to %3 step %c1 -> index {
    %6 = fir.convert %arg0 : (index) -> i32
    fir.store %6 to %0 : !fir.ref<i32>
    %c1_0 = arith.constant 1 : index
    %7 = fir.load %0 : !fir.ref<i32>
    %8 = fir.convert %7 : (i32) -> i64
    %c1_i64 = arith.constant 1 : i64
    %9 = arith.subi %8, %c1_i64 : i64
    %10 = fir.coordinate_of %1, %9 : (!fir.ref<!fir.array<10x!fir.type<_QTd{i:!fir.array<10xi32>}>>>, i64) -> !fir.ref<!fir.type<_QTd{i:!fir.array<10xi32>}>>
    %11 = fir.field_index i, !fir.type<_QTd{i:!fir.array<10xi32>}>
    %12 = fir.coordinate_of %10, %11 : (!fir.ref<!fir.type<_QTd{i:!fir.array<10xi32>}>>, !fir.field) -> !fir.ref<!fir.array<10xi32>>
    %c10 = arith.constant 10 : index
    %13 = arith.addi %c1_0, %c10 : index
    %14 = arith.subi %13, %c1_0 : index
    %c1_i64_1 = arith.constant 1 : i64
    %15 = fir.shape %c10 : (index) -> !fir.shape<1>
    %16 = fir.slice %c1_0, %14, %c1_i64_1 : (index, index, i64) -> !fir.slice<1>
    %17 = fir.array_load %12(%15) [%16] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
    %c10_i64 = arith.constant 10 : i64
    %18 = fir.convert %c10_i64 : (i64) -> index
    %c0_i32 = arith.constant 0 : i32
    %c1_2 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %19 = arith.subi %18, %c1_2 : index
    %20 = fir.do_loop %arg1 = %c0 to %19 step %c1_2 unordered iter_args(%arg2 = %17) -> (!fir.array<10xi32>) {
      %22 = fir.array_update %arg2, %c0_i32, %arg1 : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
      fir.result %22 : !fir.array<10xi32>
    }
    fir.array_merge_store %17, %20 to %12[%16] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>, !fir.slice<1>
    %21 = arith.addi %arg0, %c1 : index
    fir.result %21 : index
  }
  %5 = fir.convert %4 : (index) -> i32
  fir.store %5 to %0 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func @array_of_types() {
// CHECK:         %{{.*}} = fir.do_loop %{{.*}} = %{{.*}}  to %{{.*}}  step %{{.*}}  -> index {
// CHECK:           %{{.*}} = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%arg2 = %17) -> (!fir.array<10xi32>) {
// CHECK-NOT:         %{{.*}} = fir.array_update
// CHECK:             %[[COOR0:.*]] = fir.array_coor %{{.*}}(%{{.*}}) [%{{.*}}] %{{.*}} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:             fir.store %{{.*}} to %[[COOR0]] : !fir.ref<i32>
// CHECK-NOT:         fir.array_merge_store

// -----

// Test fir.array_load/boxed array
func.func @conversion_with_temporary_boxed_array(%arr0 : !fir.box<!fir.array<10xi32>>) {
  %c10 = arith.constant 10 : index
  %1:3 = fir.box_dims %arr0, %c10 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %shift = fir.shift %1#0 : (index) -> !fir.shift<1>
  %2 = fir.array_load %arr0(%shift) : (!fir.box<!fir.array<10xi32>>, !fir.shift<1>) -> !fir.array<10xi32>
  %c10_i64 = arith.constant 10 : i64
  %3 = fir.convert %c10_i64 : (i64) -> index
  %c1_i64 = arith.constant 1 : i64
  %c-1_i64 = arith.constant -1 : i64
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5 = fir.slice %c10_i64, %c1_i64, %c-1_i64 : (i64, i64, i64) -> !fir.slice<1>
  %6 = fir.array_load %arr0(%4) [%5] : (!fir.box<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %7 = arith.subi %3, %c1 : index
  %8 = fir.do_loop %arg0 = %c0 to %7 step %c1 unordered iter_args(%arg1 = %2) -> (!fir.array<10xi32>) {
    %9 = fir.array_fetch %6, %arg0 : (!fir.array<10xi32>, index) -> i32
    %10 = fir.array_update %arg1, %9, %arg0 : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
    fir.result %10 : !fir.array<10xi32>
  }
  fir.array_merge_store %2, %8 to %arr0 : !fir.array<10xi32>, !fir.array<10xi32>, !fir.box<!fir.array<10xi32>>
  return
}

// CHECK-LABEL: func @conversion_with_temporary_boxed_array(
// CHECK-SAME:                                              %[[ARR0:.*]]: !fir.box<!fir.array<10xi32>>)
// Allocation of temporary array.
// CHECK:         %[[TEMP:.*]] = fir.allocmem !fir.array<10xi32>
// Copy of original array to temp.
// CHECK:         fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}} : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:           %[[COOR1:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:           %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<i32>
// CHECK:           fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:         }
// Perform the assignment i = i(10:1:-1) using the temporary array.
// CHECK:         %{{.*}} = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<10xi32>) {
// CHECK-NOT:       %{{.*}} = fir.array_fetch
// CHECK-NOT:       %{{.*}} = fir.update
// CHECK:           %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) [%{{.*}}] %{{.*}} : (!fir.box<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           %[[LOAD0:.*]] = fir.load %[[COOR0]] : !fir.ref<i32>
// CHECK:           %[[COOR1:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:           fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:           fir.result %{{.*}} : !fir.array<10xi32>
// CHECK:         }
// Copy the result back to the original array.
// CHECK:         fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           %[[COOR0:.*]] = fir.array_coor %[[TEMP]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:           %[[COOR1:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}} : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:           %[[LOAD0:.*]] = fir.load %[[COOR0:.*]] : !fir.ref<i32>
// CHECK:           fir.store %[[LOAD0]] to %[[COOR1]] : !fir.ref<i32>
// CHECK:         }
// Free temporary array.
// CHECK:         fir.freemem %[[TEMP]] : !fir.heap<!fir.array<10xi32>>

// -----

// Test simple fir.array_update with Fortran.offsets attribute.
func.func @array_update_conversion(%arr1 : !fir.box<!fir.array<?x?xf32>>, %m: index, %n: index) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 2.0 : f32
  %s = fir.shape %m, %n : (index, index) -> !fir.shape<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  %av2 = fir.array_update %av1, %f, %c1, %c1 {Fortran.offsets} : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  return
}

// CHECK-LABEL: func @array_update_conversion
// CHECK-NOT:     fir.array_update
// CHECK-NOT:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:         %[[ARRAY_COOR:.*]] = fir.array_coor{{.*}}-> !fir.ref<f32>
// CHECK:         fir.store %{{.*}} to %[[ARRAY_COOR]] : !fir.ref<f32>

// -----

// Test fir.array_fetch on derived type members in an array of derived types.
func.func @array_fetch_derived_type(%0 : !fir.ref<!fir.array<10x!fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>>>) {
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QEi"}
  %c1_i32 = arith.constant 1 : i32
  %2 = fir.convert %c1_i32 : (i32) -> index
  %c10_i32 = arith.constant 10 : i32
  %3 = fir.convert %c10_i32 : (i32) -> index
  %c1 = arith.constant 1 : index
  %shape = fir.shape %2 : (index) -> !fir.shape<1>
  %arr0 = fir.array_load %0(%shape) : (!fir.ref<!fir.array<10x!fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>>
  %4 = fir.do_loop %arg0 = %2 to %3 step %c1 -> index {
    %6 = fir.convert %arg0 : (index) -> i32
    fir.store %6 to %1 : !fir.ref<i32>
    %c1_i32_0 = arith.constant 1 : i32
    %7 = fir.load %1 : !fir.ref<i32>
    %8 = fir.convert %7 : (i32) -> i64
    %c1_i64 = arith.constant 1 : i64
    %9 = arith.subi %8, %c1_i64 : i64
    %11 = fir.field_index mt, !fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>
    %12 = fir.field_index mem, !fir.type<_QTt{mem:i32}>
    %idx = fir.convert %9 : (i64) -> index
    %res = fir.array_fetch %arr0, %idx, %11, %12 : (!fir.array<10x!fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>>, index, !fir.field, !fir.field) -> i32
    %14 = arith.addi %arg0, %c1 : index
    fir.result %14 : index
  }
  %5 = fir.convert %4 : (index) -> i32
  fir.store %5 to %1 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func @array_fetch_derived_type(
// CHECK-SAME:                                 %[[ARR0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>>>) {
// CHECK:         %{{.*}} = fir.do_loop
// CHECK:           %[[FIELD_MT:.*]] = fir.field_index mt, !fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>
// CHECK:           %[[FIELD_MEM:.*]] = fir.field_index mem, !fir.type<_QTt{mem:i32}>
// CHECK-NOT:       %{{.*}} = fir.array_fetch
// CHECK:           %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<10x!fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>>
// CHECK:           %[[COOR_OF:.*]] = fir.coordinate_of %[[COOR0]], %[[FIELD_MT]], %[[FIELD_MEM]] : (!fir.ref<!fir.type<_QTu{mt:!fir.type<_QTt{mem:i32}>}>>, !fir.field, !fir.field) -> !fir.ref<i32>
// CHECK:           %{{.*}} = fir.load %[[COOR_OF]] : !fir.ref<i32>

// -----

// Test simple fir.array_load/fir.array_update conversion without copy-in/copy-out with a `fir.box`
func.func @array_update_conversion(%arr1 : !fir.box<!fir.array<?x?xf32>>, %m: index, %n: index) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 2.0 : f32
  %s = fir.shape %m, %n : (index, index) -> !fir.shape<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  %av2 = fir.array_update %av1, %f, %c1, %c1 : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  return
}

// -----

// Test array operation with conditional update.

func.func @array_operation_with_cond_update(%arg0: !fir.ref<!fir.array<100xf32>>, %cond1: i1) {
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %f = arith.constant 2.0 : f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %arg2 = fir.if %cond1 -> !fir.array<100xf32> {
    fir.result %2 : !fir.array<100xf32>
  } else {
    %r = fir.array_update %2, %f, %c1 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %r : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %arg2 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}

// CHECK-LABEL: func @array_operation_with_cond_update(
// CHECK-SAME:                                         %[[ARG0:.*]]: !fir.ref<!fir.array<100xf32>>, %[[COND:.*]]: i1) {
// CHECK:         %[[ARRAY_LOAD:.*]] = fir.undefined !fir.array<100xf32>
// CHECK:         %[[IF_RES:.*]] = fir.if %[[COND]] -> (!fir.array<100xf32>) {
// CHECK:           fir.result %[[ARRAY_LOAD]] : !fir.array<100xf32>
// CHECK:         } else {
// CHECK:           %[[UPDATE0:.*]] = fir.array_coor %[[ARG0]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:           fir.store %{{.*}} to %{{.*}} : !fir.ref<f32>
// CHECK:           fir.result %[[ARRAY_LOAD]] : !fir.array<100xf32>
// CHECK:    }
