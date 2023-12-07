// RUN: fir-opt --inline-elementals %s | FileCheck %s

// check inlining one elemental into another
// a = b * c + d
func.func @inline_to_elemental(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"}, %arg3: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "c"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3:2 = hlfir.declare %arg3 {uniq_name = "d"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %c0 = arith.constant 0 : index
  %4:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg4: index):
    %8 = hlfir.designate %1#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %9 = hlfir.designate %2#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %10 = fir.load %8 : !fir.ref<i32>
    %11 = fir.load %9 : !fir.ref<i32>
    %12 = arith.muli %10, %11 : i32
    hlfir.yield_element %12 : i32
  }
  %7 = hlfir.elemental %5 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg4: index):
    %8 = hlfir.apply %6, %arg4 : (!hlfir.expr<?xi32>, index) -> i32
    %9 = hlfir.designate %3#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %10 = fir.load %9 : !fir.ref<i32>
    %11 = arith.addi %8, %10 : i32
    hlfir.yield_element %11 : i32
  }
  hlfir.assign %7 to %0#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %7 : !hlfir.expr<?xi32>
  hlfir.destroy %6 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @inline_to_elemental
// CHECK-SAME:      %[[A_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}
// CHECK-SAME:      %[[B_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}
// CHECK-SAME:      %[[C_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"}
// CHECK-SAME:      %[[D_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "d"}
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
// CHECK-DAG:     %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
// CHECK-DAG:     %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
// CHECK-DAG:     %[[D:.*]]:2 = hlfir.declare %[[D_ARG]]
// CHECK-NEXT:    %[[B_DIM0:.*]]:3 = fir.box_dims %[[B]]#0, %[[C0]]
// CHECK-NEXT:    %[[B_SHAPE:.*]] = fir.shape %[[B_DIM0]]#1
// CHECK-NEXT:    %[[EXPR:.*]] = hlfir.elemental %[[B_SHAPE]]
// CHECK-NEXT:    ^bb0(%[[I:.*]]: index):
// inline the first elemental:
// CHECK-NEXT:      %[[B_I_REF:.*]] = hlfir.designate %[[B]]#0 (%[[I]])
// CHECK-NEXT:      %[[C_I_REF:.*]] = hlfir.designate %[[C]]#0 (%[[I]])
// CHECK-NEXT:      %[[B_I:.*]] = fir.load %[[B_I_REF]]
// CHECK-NEXT:      %[[C_I:.*]] = fir.load %[[C_I_REF]]
// CHECK-NEXT:      %[[MUL:.*]] = arith.muli %[[B_I]], %[[C_I]]
// second elemental:
// CHECK-NEXT:      %[[D_I_REF:.*]] = hlfir.designate %[[D]]#0 (%[[I]])
// CHECK-NEXT:      %[[D_I:.*]] = fir.load %[[D_I_REF]]
// CHECK-NEXT:      %[[ADD:.*]] = arith.addi %[[MUL]], %[[D_I]]
// CHECK-NEXT:      hlfir.yield_element %[[ADD]]
// CHECK-NEXT:    }
// CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[A]]#0
// CHECK-NEXT:    hlfir.destroy %[[EXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// check inlining into a do_loop
func.func @inline_to_loop(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"}, %arg3: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "c"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3:2 = hlfir.declare %arg3 {uniq_name = "d"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %c0 = arith.constant 0 : index
  %4:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg4: index):
    %8 = hlfir.designate %1#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %9 = hlfir.designate %2#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %10 = fir.load %8 : !fir.ref<i32>
    %11 = fir.load %9 : !fir.ref<i32>
    %12 = arith.muli %10, %11 : i32
    hlfir.yield_element %12 : i32
  }
  %array = fir.array_load %0#0 : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
  %c1 = arith.constant 1 : index
  %max = arith.subi %4#1, %c1 : index
  %7 = fir.do_loop %arg4 = %c0 to %max step %c1 unordered iter_args(%arg5 = %array) -> (!fir.array<?xi32>) {
    %8 = hlfir.apply %6, %arg4 : (!hlfir.expr<?xi32>, index) -> i32
    %9 = hlfir.designate %3#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %10 = fir.load %9 : !fir.ref<i32>
    %11 = arith.addi %8, %10 : i32
    %12 = fir.array_update %arg5, %11, %arg4 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
    fir.result %12 : !fir.array<?xi32>
  }
  fir.array_merge_store %array, %7 to %arg0 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %6 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @inline_to_loop
// CHECK-SAME:      %[[A_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}
// CHECK-SAME:      %[[B_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}
// CHECK-SAME:      %[[C_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"}
// CHECK-SAME:      %[[D_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "d"}
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
// CHECK-DAG:     %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
// CHECK-DAG:     %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
// CHECK-DAG:     %[[D:.*]]:2 = hlfir.declare %[[D_ARG]]
// CHECK-NEXT:    %[[B_DIM0:.*]]:3 = fir.box_dims %[[B]]#0, %[[C0]]
// CHECK-NEXT:    %[[ARRAY:.*]] = fir.array_load %[[A]]#0
// CHECK-NEXT:    %[[MAX:.*]] = arith.subi %[[B_DIM0]]#1, %[[C1]]
// CHECK-NEXT:    %[[LOOP:.*]] = fir.do_loop %[[I:.*]] = %[[C0]] to %[[MAX]] step %[[C1]] unordered iter_args(%[[LOOP_ARRAY:.*]] = %[[ARRAY]])
// inline the elemental:
// CHECK-NEXT:      %[[B_I_REF:.*]] = hlfir.designate %[[B]]#0 (%[[I]])
// CHECK-NEXT:      %[[C_I_REF:.*]] = hlfir.designate %[[C]]#0 (%[[I]])
// CHECK-NEXT:      %[[B_I:.*]] = fir.load %[[B_I_REF]]
// CHECK-NEXT:      %[[C_I:.*]] = fir.load %[[C_I_REF]]
// CHECK-NEXT:      %[[MUL:.*]] = arith.muli %[[B_I]], %[[C_I]]
// loop body:
// CHECK-NEXT:      %[[D_I_REF:.*]] = hlfir.designate %[[D]]#0 (%[[I]])
// CHECK-NEXT:      %[[D_I:.*]] = fir.load %[[D_I_REF]]
// CHECK-NEXT:      %[[ADD:.*]] = arith.addi %[[MUL]], %[[D_I]]
// CHECK-NEXT:      %[[ARRAY_UPD:.*]] = fir.array_update %[[LOOP_ARRAY]], %[[ADD]], %[[I]]
// CHECK-NEXT:      fir.result %[[ARRAY_UPD]]
// CHECK-NEXT:    }
// CHECK-NEXT:    fir.array_merge_store %[[ARRAY]], %[[LOOP]] to %[[A_ARG]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// inlining into a single hlfir.apply
// a = (b * c)[1]
func.func @inline_to_apply(%arg0: !fir.ref<i32> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "c"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %c0 = arith.constant 0 : index
  %4:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg4: index):
    %8 = hlfir.designate %1#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %9 = hlfir.designate %2#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %10 = fir.load %8 : !fir.ref<i32>
    %11 = fir.load %9 : !fir.ref<i32>
    %12 = arith.muli %10, %11 : i32
    hlfir.yield_element %12 : i32
  }
  %c1 = arith.constant 1 : index
  %res = hlfir.apply %6, %c1 : (!hlfir.expr<?xi32>, index) -> i32
  fir.store %res to %0#0 : !fir.ref<i32>
  hlfir.destroy %6 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @inline_to_apply
// CHECK-SAME:      %[[A_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}
// CHECK-SAME:      %[[B_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}
// CHECK-SAME:      %[[C_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"}
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
// CHECK-DAG:     %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
// CHECK-DAG:     %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
// inline the elemental:
// CHECK-NEXT:    %[[B_1_REF:.*]] = hlfir.designate %[[B]]#0 (%[[C1]])
// CHECK-NEXT:    %[[C_1_REF:.*]] = hlfir.designate %[[C]]#0 (%[[C1]])
// CHECK-NEXT:    %[[B_1:.*]] = fir.load %[[B_1_REF]]
// CHECK-NEXT:    %[[C_1:.*]] = fir.load %[[C_1_REF]]
// CHECK-NEXT:    %[[MUL:.*]] = arith.muli %[[B_1]], %[[C_1]]
// store:
// CHECK-NEXT:    fir.store %[[MUL]] to %0#0 : !fir.ref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// Check long chains of elementals
// subroutine reproducer(a)
//   real, dimension(:) :: a
//   a = sqrt(a * (a - 1))
// end subroutine
func.func @_QPreproducer(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFreproducerEa"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %1:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %2 = fir.shape %1#1 : (index) -> !fir.shape<1>
  %3 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%arg1: index):
    %9 = hlfir.designate %0#0 (%arg1)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %10 = fir.load %9 : !fir.ref<f32>
    %11 = arith.subf %10, %cst fastmath<contract> : f32
    hlfir.yield_element %11 : f32
  }
  %4 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%arg1: index):
    %9 = hlfir.apply %3, %arg1 : (!hlfir.expr<?xf32>, index) -> f32
    %10 = hlfir.no_reassoc %9 : f32
    hlfir.yield_element %10 : f32
  }
  %c0_0 = arith.constant 0 : index
  %5:3 = fir.box_dims %0#0, %c0_0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %6 = fir.shape %5#1 : (index) -> !fir.shape<1>
  %7 = hlfir.elemental %6 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%arg1: index):
    %9 = hlfir.designate %0#0 (%arg1)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %10 = hlfir.apply %4, %arg1 : (!hlfir.expr<?xf32>, index) -> f32
    %11 = fir.load %9 : !fir.ref<f32>
    %12 = arith.mulf %11, %10 fastmath<contract> : f32
    hlfir.yield_element %12 : f32
  }
  %8 = hlfir.elemental %6 : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%arg1: index):
    %9 = hlfir.apply %7, %arg1 : (!hlfir.expr<?xf32>, index) -> f32
    %10 = math.sqrt %9 fastmath<contract> : f32
    hlfir.yield_element %10 : f32
  }
  hlfir.assign %8 to %0#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %8 : !hlfir.expr<?xf32>
  hlfir.destroy %7 : !hlfir.expr<?xf32>
  hlfir.destroy %4 : !hlfir.expr<?xf32>
  hlfir.destroy %3 : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL: func.func @_QPreproducer
// CHECK-SAME:      %[[A_ARG:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}
// CHECK-DAG:     %[[CST:.*]] = arith.constant 1.0000
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0
// CHECK-DAG:     %[[A_VAR:.*]]:2 = hlfir.declare %[[A_ARG]]
// CHECK-NEXT:    %[[A_DIMS_0:.*]]:3 = fir.box_dims %[[A_VAR]]#0, %[[C0]]
// CHECK-NEXT:    %[[SHAPE_0:.*]] = fir.shape %[[A_DIMS_0]]#1
// all in one elemental:
// CHECK-NEXT:    %[[EXPR:.*]] = hlfir.elemental %[[SHAPE_0]]
// CHECK-NEXT:    ^bb0(%[[I:.*]]: index):
// CHECK-NEXT:      %[[A_I0:.*]] = hlfir.designate %[[A_VAR]]#0 (%[[I]])
// CHECK-NEXT:      %[[A_I1:.*]] = hlfir.designate %[[A_VAR]]#0 (%[[I]])
// CHECK-NEXT:      %[[A_I1_VAL:.*]] = fir.load %[[A_I1]]
// CHECK-NEXT:      %[[SUB:.*]] = arith.subf %[[A_I1_VAL]], %[[CST]]
// CHECK-NEXT:      %[[SUB0:.*]] = hlfir.no_reassoc %[[SUB]] : f32
// CHECK-NEXT:      %[[A_I0_VAL:.*]] = fir.load %[[A_I0]]
// CHECK-NEXT:      %[[MUL:.*]] = arith.mulf %[[A_I0_VAL]], %[[SUB0]]
// CHECK-NEXT:      %[[SQRT:.*]] = math.sqrt %[[MUL]]
// CHECK-NEXT:      hlfir.yield_element %[[SQRT]]
// CHECK-NEXT:    }
// CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[A_VAR]]#0
// CHECK-NEXT:    hlfir.destroy %[[EXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// Check that the ordered elemental is not inlined into another:
// a = b + c + d
func.func @noinline_ordered(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"}, %arg3: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "c"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3:2 = hlfir.declare %arg3 {uniq_name = "d"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %c0 = arith.constant 0 : index
  %4:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg4: index):
    %8 = hlfir.designate %1#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %9 = hlfir.designate %2#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %10 = fir.load %8 : !fir.ref<i32>
    %11 = fir.load %9 : !fir.ref<i32>
    %12 = arith.muli %10, %11 : i32
    hlfir.yield_element %12 : i32
  }
  %7 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg4: index):
    %8 = hlfir.apply %6, %arg4 : (!hlfir.expr<?xi32>, index) -> i32
    %9 = hlfir.designate %3#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %10 = fir.load %9 : !fir.ref<i32>
    %11 = arith.addi %8, %10 : i32
    hlfir.yield_element %11 : i32
  }
  hlfir.assign %7 to %0#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %7 : !hlfir.expr<?xi32>
  hlfir.destroy %6 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL:   func.func @noinline_ordered(
// CHECK-SAME:      %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:      %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"},
// CHECK-SAME:      %[[VAL_2:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"},
// CHECK-SAME:      %[[VAL_3:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "d"}) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "b"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "c"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "d"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_6]]#0, %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_11:.*]] = hlfir.elemental %[[VAL_10]] : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index):
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_12]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_12]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
// CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_15]], %[[VAL_16]] : i32
// CHECK:             hlfir.yield_element %[[VAL_17]] : i32
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = hlfir.elemental %[[VAL_10]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
// CHECK:           ^bb0(%[[VAL_19:.*]]: index):
// CHECK:             %[[VAL_20:.*]] = hlfir.apply %[[VAL_21:.*]], %[[VAL_19]] : (!hlfir.expr<?xi32>, index) -> i32
// CHECK:             %[[VAL_22:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_19]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<i32>
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_20]], %[[VAL_23]] : i32
// CHECK:             hlfir.yield_element %[[VAL_24]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_25:.*]] to %[[VAL_5]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
// CHECK:           hlfir.destroy %[[VAL_25]] : !hlfir.expr<?xi32>
// CHECK:           hlfir.destroy %[[VAL_26:.*]] : !hlfir.expr<?xi32>
// CHECK:           return
// CHECK:         }
