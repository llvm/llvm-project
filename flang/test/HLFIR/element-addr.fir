// Test hlfir.element_addr operation parse, verify (no errors), and unparse.
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test_element_addr(%x: !fir.ref<!fir.array<100xf32>>, %vector: !fir.ref<!fir.array<20xi32>>, %y: !fir.ref<!fir.array<20xf32>>) {
  %c20 = arith.constant 20 : index
  %vector_shape = fir.shape %c20 : (index) -> !fir.shape<1>
  hlfir.region_assign {
    hlfir.yield %y : !fir.ref<!fir.array<20xf32>>
  } to {
    hlfir.elemental_addr %vector_shape  : !fir.shape<1> {
      ^bb0(%i: index):
      %0 = hlfir.designate %vector (%i)  : (!fir.ref<!fir.array<20xi32>>, index) -> !fir.ref<i32>
      %1 = fir.load %0 : !fir.ref<i32>
      %x_element_addr = hlfir.designate %x (%1)  : (!fir.ref<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
      hlfir.yield %x_element_addr : !fir.ref<f32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_element_addr(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.ref<!fir.array<20xi32>>,
// CHECK-SAME:                                 %[[VAL_2:.*]]: !fir.ref<!fir.array<20xf32>>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           hlfir.region_assign {
// CHECK:             hlfir.yield %[[VAL_2]] : !fir.ref<!fir.array<20xf32>>
// CHECK:           } to {
// CHECK:             hlfir.elemental_addr %[[VAL_4]] : !fir.shape<1> {
// CHECK:             ^bb0(%[[VAL_5:.*]]: index):
// CHECK:               %[[VAL_6:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_5]])  : (!fir.ref<!fir.array<20xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
// CHECK:               %[[VAL_8:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_7]])  : (!fir.ref<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
// CHECK:               hlfir.yield %[[VAL_8]] : !fir.ref<f32>
// CHECK:             }
// CHECK:           }



func.func @test_element_addr_cleanup(%x: !fir.box<!fir.array<?x!fir.char<1,?>>>, %y: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
  hlfir.region_assign {
    hlfir.yield %y : !fir.box<!fir.array<?x!fir.char<1,?>>>
  } to {
    %c0 = arith.constant 0 : index
    %len = fir.box_elesize %x : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
    %vector = fir.call @returns_alloc() : () -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    %vector_dim:3 = fir.box_dims %x, %c0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
    %vector_shape = fir.shape %vector_dim#1 : (index) -> !fir.shape<1>
    hlfir.elemental_addr %vector_shape  typeparams %len : !fir.shape<1>, index {
      ^bb0(%i: index):
      %0 = hlfir.designate %vector (%i)  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
      %1 = fir.load %0 : !fir.ref<i32>
      %x_element_addr = hlfir.designate %x (%1) typeparams %len  : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i32, index) -> !fir.boxchar<1>
      hlfir.yield %x_element_addr : !fir.boxchar<1>
    } cleanup {
      %addr = fir.box_addr %vector : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
      fir.freemem %addr : !fir.heap<!fir.array<?xi32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_element_addr_cleanup(
// CHECK-SAME:                                         %[[VAL_0:[^:]*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
// CHECK:           hlfir.region_assign {
// CHECK:             hlfir.yield %[[VAL_1]] : !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           } to {
// CHECK:             %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_3:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_4:.*]] = fir.call @returns_alloc() : () -> !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK:             %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
// CHECK:             hlfir.elemental_addr %[[VAL_6]] typeparams %[[VAL_3]] : !fir.shape<1>, index {
// CHECK:             ^bb0(%[[VAL_7:.*]]: index):
// CHECK:               %[[VAL_8:.*]] = hlfir.designate %[[VAL_4]] (%[[VAL_7]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:               %[[VAL_10:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_9]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i32, index) -> !fir.boxchar<1>
// CHECK:               hlfir.yield %[[VAL_10]] : !fir.boxchar<1>
// CHECK:             } cleanup {
// CHECK:               %[[VAL_11:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK:               fir.freemem %[[VAL_11]] : !fir.heap<!fir.array<?xi32>>
// CHECK:             }
// CHECK:           }

func.func @unordered() {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  hlfir.region_assign {
    %addr = fir.undefined !fir.ref<!fir.array<10x20xf32>>
    hlfir.yield %addr : !fir.ref<!fir.array<10x20xf32>>
  } to {
    hlfir.elemental_addr %0 unordered : !fir.shape<2> {
    ^bb0(%i: index, %j: index):
      %addr = fir.undefined !fir.ref<f32>
      hlfir.yield %addr : !fir.ref<f32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @unordered() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
// CHECK:           hlfir.region_assign {
// CHECK:             %[[VAL_3:.*]] = fir.undefined !fir.ref<!fir.array<10x20xf32>>
// CHECK:             hlfir.yield %[[VAL_3]] : !fir.ref<!fir.array<10x20xf32>>
// CHECK:           } to {
// CHECK:             hlfir.elemental_addr %[[VAL_2]] unordered : !fir.shape<2> {
// CHECK:             ^bb0(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
// CHECK:               %[[VAL_6:.*]] = fir.undefined !fir.ref<f32>
// CHECK:               hlfir.yield %[[VAL_6]] : !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// "X(VECTOR) = Y" with polymorphic X and Y and user defined assignment.
func.func @test_mold(%x: !fir.class<!fir.array<?x!fir.type<t>>>, %y: !fir.class<!fir.array<?x!fir.type<t>>>, %vector: !fir.box<!fir.array<?xi64>>) {
  hlfir.region_assign {
    hlfir.yield %y : !fir.class<!fir.array<?x!fir.type<t>>>
  } to {
    %c0 = arith.constant 0 : index
    %0:3 = fir.box_dims %vector, %c0 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %1 = fir.shape %0#1 : (index) -> !fir.shape<1>
    hlfir.elemental_addr %1 mold %x unordered : !fir.shape<1>, !fir.class<!fir.array<?x!fir.type<t>>> {
    ^bb0(%arg3: index):
      %2 = hlfir.designate %vector (%arg3)  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
      %3 = fir.load %2 : !fir.ref<i64>
      %4 = hlfir.designate %x (%3)  : (!fir.class<!fir.array<?x!fir.type<t>>>, i64) -> !fir.class<!fir.type<t>>
      hlfir.yield %4 : !fir.class<!fir.type<t>>
    }
  } user_defined_assign  (%arg3: !fir.class<!fir.type<t>>) to (%arg4: !fir.class<!fir.type<t>>) {
    fir.call @user_def_assign(%arg4, %arg3) : (!fir.class<!fir.type<t>>, !fir.class<!fir.type<t>>) -> ()
  }
  return
}
func.func private @user_def_assign(!fir.class<!fir.type<t>>, !fir.class<!fir.type<t>>)
// CHECK-LABEL: func.func @test_mold(
// CHECK-SAME:                       %[[VAL_0:[^:]*]]: !fir.class<!fir.array<?x!fir.type<t>>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: !fir.class<!fir.array<?x!fir.type<t>>>,
// CHECK-SAME:                       %[[VAL_2:.*]]: !fir.box<!fir.array<?xi64>>) {
// CHECK:         hlfir.region_assign {
// CHECK:           hlfir.yield %[[VAL_1]] : !fir.class<!fir.array<?x!fir.type<t>>>
// CHECK:         } to {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
// CHECK:           hlfir.elemental_addr %[[VAL_5]] mold %[[VAL_0]] unordered : !fir.shape<1>, !fir.class<!fir.array<?x!fir.type<t>>> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_2]] (%[[VAL_6]])  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
// CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i64>
// CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_8]])  : (!fir.class<!fir.array<?x!fir.type<t>>>, i64) -> !fir.class<!fir.type<t>>
// CHECK:             hlfir.yield %[[VAL_9]] : !fir.class<!fir.type<t>>
// CHECK:           }
// CHECK:         } user_defined_assign  (%[[VAL_10:.*]]: !fir.class<!fir.type<t>>) to (%[[VAL_11:.*]]: !fir.class<!fir.type<t>>) {
// CHECK:           fir.call @user_def_assign(%[[VAL_11]], %[[VAL_10]]) : (!fir.class<!fir.type<t>>, !fir.class<!fir.type<t>>) -> ()
// CHECK:         }
