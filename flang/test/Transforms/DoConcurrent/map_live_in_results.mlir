// Tests that every value used inside a device-mapped `do concurrent` loop is
// mapped to the target region, including:
// * multiple results of a non-declare op (`fir.box_dims`), and
// * both results of an `hlfir.declare`, regardless of which one the loop
//   uses first.

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=device" %s -o - | FileCheck %s

func.func @box_dims_results(%box: !fir.box<!fir.array<?xf32>>) {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %box, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_concurrent {
    %0 = fir.alloca i32 {bindc_name = "i"}
    %1:2 = hlfir.declare %0 uniq_name("_QFEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1) {
      %2 = fir.convert %arg0 : (index) -> i32
      fir.store %2 to %1#0 : !fir.ref<i32>
      %3 = arith.addi %dims#0, %dims#1 : index
      %4 = fir.convert %3 : (index) -> i32
      fir.store %4 to %1#0 : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL: func.func @box_dims_results
// CHECK:         %[[DIMS:.*]]:3 = fir.box_dims
// CHECK-DAG:     fir.store %[[DIMS]]#0 to %[[LB_TMP:.*]] : !fir.ref<index>
// CHECK-DAG:     fir.store %[[DIMS]]#1 to %[[EXT_TMP:.*]] : !fir.ref<index>
// CHECK-DAG:     %[[LB_MAP:.*]] = omp.map.info var_ptr(%[[LB_TMP]] : !fir.ref<index>, index)
// CHECK-DAG:     %[[EXT_MAP:.*]] = omp.map.info var_ptr(%[[EXT_TMP]] : !fir.ref<index>, index)
// CHECK:         omp.target
// CHECK-SAME:      %[[LB_MAP]] -> %[[LB_ARG:arg[0-9]+]], %[[EXT_MAP]] -> %[[EXT_ARG:arg[0-9]+]]
// CHECK-DAG:       %[[LB_DECL:.*]]:2 = hlfir.declare %[[LB_ARG]]
// CHECK-DAG:       %[[EXT_DECL:.*]]:2 = hlfir.declare %[[EXT_ARG]]
// CHECK-DAG:       %[[LB:.*]] = fir.load %[[LB_DECL]]#1 : !fir.ref<index>
// CHECK-DAG:       %[[EXT:.*]] = fir.load %[[EXT_DECL]]#1 : !fir.ref<index>
// CHECK:           omp.loop_nest
// CHECK:             arith.addi %[[LB]], %[[EXT]] : index

func.func @declare_results_order(%arg: !fir.box<!fir.array<?xf32>>) {
  %a:2 = hlfir.declare %arg uniq_name("_QFEa") : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_concurrent {
    %0 = fir.alloca i32 {bindc_name = "i"}
    %1:2 = hlfir.declare %0 uniq_name("_QFEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1) {
      %2 = fir.convert %arg0 : (index) -> i32
      fir.store %2 to %1#0 : !fir.ref<i32>
      %d0:3 = fir.box_dims %a#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
      %d1:3 = fir.box_dims %a#1, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
      %3 = arith.addi %d0#1, %d1#1 : index
      %4 = fir.convert %3 : (index) -> i32
      fir.store %4 to %1#0 : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL: func.func @declare_results_order
// CHECK:         omp.target
// CHECK:           %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFEa")
// CHECK-DAG:       %[[A1:.*]] = fir.load %[[A]]#1 : !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK-DAG:       %[[A0:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK:           omp.loop_nest
// CHECK:             fir.box_dims %[[A0]], %{{.*}}
// CHECK:             fir.box_dims %[[A1]], %{{.*}}
