// Test code generation of hlfir.region_assign with user defined
// assignment.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @test_simple_scalar(%i: !fir.ref<i32>, %l: !fir.ref<!fir.logical<4>>) {
  hlfir.region_assign {
    hlfir.yield %l : !fir.ref<!fir.logical<4>>
  } to {
    hlfir.yield %i : !fir.ref<i32>
  } user_defined_assign (%arg0: !fir.ref<!fir.logical<4>>) to (%arg1: !fir.ref<i32>) {
    fir.call @logical_to_numeric(%arg1, %arg0) : (!fir.ref<i32>, !fir.ref<!fir.logical<4>>) -> ()
  }
  return
}
// CHECK-LABEL:   func.func @test_simple_scalar(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>) {
// CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
// CHECK:  %[[VAL_3:.*]]:3 = hlfir.associate %[[VAL_2]] {uniq_name = ".tmp.assign"} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
// CHECK:  fir.call @logical_to_numeric(%[[VAL_0]], %[[VAL_3]]#0) : (!fir.ref<i32>, !fir.ref<!fir.logical<4>>) -> ()
// CHECK:  hlfir.end_associate %[[VAL_3]]#1, %[[VAL_3]]#2 : !fir.ref<!fir.logical<4>>, i1

func.func @test_elemental_overlap(%i: !fir.ref<!fir.array<10xi32>>) {
  %c0_i32 = arith.constant 0 : i32
  %c10 = arith.constant 10 : index
  %shape  = fir.shape %c10 : (index) -> !fir.shape<1>
  hlfir.region_assign {
    %cmp = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
    ^bb0(%j: index):
      %ielt = hlfir.designate %i (%j)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      %ielt_val = fir.load %ielt : !fir.ref<i32>
      %smaller = arith.cmpi slt, %ielt_val, %c0_i32 : i32
      %smaller_cast = fir.convert %smaller : (i1) -> !fir.logical<4>
      hlfir.yield_element %smaller_cast : !fir.logical<4>
    }
    hlfir.yield %cmp : !hlfir.expr<10x!fir.logical<4>>
  } to {
    hlfir.yield %i : !fir.ref<!fir.array<10xi32>>
  } user_defined_assign (%arg0: !fir.logical<4>) to (%arg1: !fir.ref<i32>) {
    fir.call @logical_value_to_numeric(%arg1, %arg0) : (!fir.ref<i32>, !fir.logical<4>) -> ()
  }
  return
}
// CHECK-LABEL:   func.func @test_elemental_overlap(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_4:.*]] = hlfir.elemental %[[VAL_3]] : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
// CHECK:  ^bb0(%[[VAL_5:.*]]: index):
// CHECK:    %[[VAL_6:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_5]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:    %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
// CHECK:    %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_1]] : i32
// CHECK:    %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
// CHECK:    hlfir.yield_element %[[VAL_9]] : !fir.logical<4>
// CHECK:  }
// CHECK:  %[[VAL_10:.*]]:3 = hlfir.associate %[[VAL_11:.*]](%[[VAL_3]]) {uniq_name = ".tmp.assign"} : (!hlfir.expr<10x!fir.logical<4>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.ref<!fir.array<10x!fir.logical<4>>>, i1)
// CHECK:  %[[VAL_12:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:  fir.do_loop %[[VAL_15:.*]] = %[[VAL_14]] to %[[VAL_12]] step %[[VAL_14]] {
// CHECK:    %[[VAL_16:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_15]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:    %[[VAL_17:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_15]])  : (!fir.ref<!fir.array<10x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:    %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<!fir.logical<4>>
// CHECK:    fir.call @logical_value_to_numeric(%[[VAL_16]], %[[VAL_18]]) : (!fir.ref<i32>, !fir.logical<4>) -> ()
// CHECK:  }
// CHECK:  hlfir.end_associate %[[VAL_10]]#1, %[[VAL_10]]#2 : !fir.ref<!fir.array<10x!fir.logical<4>>>, i1

func.func @test_array_overlap(%i: !fir.ref<!fir.array<10xi32>>) {
  %c0_i32 = arith.constant 0 : i32
  %c10 = arith.constant 10 : index
  %shape  = fir.shape %c10 : (index) -> !fir.shape<1>
  hlfir.region_assign {
    %cmp = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
    ^bb0(%j: index):
      %ielt = hlfir.designate %i (%j)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      %ielt_val = fir.load %ielt : !fir.ref<i32>
      %smaller = arith.cmpi slt, %ielt_val, %c0_i32 : i32
      %smaller_cast = fir.convert %smaller : (i1) -> !fir.logical<4>
      hlfir.yield_element %smaller_cast : !fir.logical<4>
    }
    hlfir.yield %cmp : !hlfir.expr<10x!fir.logical<4>>
  } to {
    hlfir.yield %i : !fir.ref<!fir.array<10xi32>>
  } user_defined_assign (%arg0: !hlfir.expr<10x!fir.logical<4>>) to (%arg1: !fir.ref<!fir.array<10xi32>>) {
     %1:3 = hlfir.associate %arg0(%shape) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<10x!fir.logical<4>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.ref<!fir.array<10x!fir.logical<4>>>, i1)
    fir.call @logical_array_to_numeric(%arg1, %1#0) : (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10x!fir.logical<4>>>) -> ()
     hlfir.end_associate %1#1, %1#2 : !fir.ref<!fir.array<10x!fir.logical<4>>>, i1
  }
  return
}
// CHECK-LABEL:   func.func @test_array_overlap(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_4:.*]] = hlfir.elemental %[[VAL_3]] : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
// CHECK:  ^bb0(%[[VAL_5:.*]]: index):
// CHECK:    %[[VAL_6:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_5]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:    %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
// CHECK:    %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_1]] : i32
// CHECK:    %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
// CHECK:    hlfir.yield_element %[[VAL_9]] : !fir.logical<4>
// CHECK:  }
// CHECK:  %[[VAL_10:.*]]:3 = hlfir.associate %[[VAL_11:.*]](%[[VAL_3]]) {uniq_name = ".tmp.assign"} : (!hlfir.expr<10x!fir.logical<4>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.ref<!fir.array<10x!fir.logical<4>>>, i1)
// CHECK:  %[[VAL_12:.*]] = hlfir.as_expr %[[VAL_10]]#0 : (!fir.ref<!fir.array<10x!fir.logical<4>>>) -> !hlfir.expr<10x!fir.logical<4>>
// CHECK:  %[[VAL_13:.*]]:3 = hlfir.associate %[[VAL_12]](%[[VAL_3]]) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<10x!fir.logical<4>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.ref<!fir.array<10x!fir.logical<4>>>, i1)
// CHECK:  fir.call @logical_array_to_numeric(%[[VAL_0]], %[[VAL_13]]#0) : (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10x!fir.logical<4>>>) -> ()
// CHECK:  hlfir.end_associate %[[VAL_13]]#1, %[[VAL_13]]#2 : !fir.ref<!fir.array<10x!fir.logical<4>>>, i1
// CHECK:  hlfir.destroy %[[VAL_12]] : !hlfir.expr<10x!fir.logical<4>>
// CHECK:  hlfir.end_associate %[[VAL_10]]#1, %[[VAL_10]]#2 : !fir.ref<!fir.array<10x!fir.logical<4>>>, i1

func.func @test_scalar_forall_overlap(%i: !fir.ref<!fir.array<10xi32>>) {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c10 : index
  }  (%j: index) {
    hlfir.region_assign {
      %reverse_j = arith.subi %c11, %j : index
      %ielt = hlfir.designate %i (%reverse_j)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      %ielt_val = fir.load %ielt : !fir.ref<i32>
      %smaller = arith.cmpi slt, %ielt_val, %c0_i32 : i32
      hlfir.yield %smaller : i1
    } to {
      %ielt = hlfir.designate %i (%j)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      hlfir.yield %ielt : !fir.ref<i32>
    } user_defined_assign (%arg0: i1) to (%arg1: !fir.ref<i32>) {
      %cast = fir.convert %arg0 : (i1) -> !fir.logical<4>
      fir.call @logical_value_to_numeric(%arg1, %cast) : (!fir.ref<i32>, !fir.logical<4>) -> ()
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_scalar_forall_overlap(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) {
// CHECK:  %[[VAL_1:.*]] = fir.alloca index
// CHECK:  %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_5:.*]] = arith.constant 11 : index
// CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_8:.*]] = arith.subi %[[VAL_4]], %[[VAL_3]] : index
// CHECK:  %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_6]] : index
// CHECK:  %[[VAL_10:.*]] = arith.divsi %[[VAL_9]], %[[VAL_6]] : index
// CHECK:  %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_7]] : index
// CHECK:  %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_7]] : index
// CHECK:  %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:  fir.store %[[VAL_13]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:  %[[VAL_15:.*]] = fir.allocmem !fir.array<?xi1>, %[[VAL_12]] {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:  %[[VAL_16:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_15]](%[[VAL_16]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi1>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi1>>, !fir.heap<!fir.array<?xi1>>)
// CHECK:  fir.do_loop %[[VAL_18:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:    %[[VAL_19:.*]] = arith.subi %[[VAL_5]], %[[VAL_18]] : index
// CHECK:    %[[VAL_20:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_19]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:    %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i32>
// CHECK:    %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_2]] : i32
// CHECK:    %[[VAL_23:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
// CHECK:    %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_14]] : index
// CHECK:    fir.store %[[VAL_24]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:    %[[VAL_25:.*]] = hlfir.designate %[[VAL_17]]#0 (%[[VAL_23]])  : (!fir.box<!fir.array<?xi1>>, index) -> !fir.ref<i1>
// CHECK:    hlfir.assign %[[VAL_22]] to %[[VAL_25]] : i1, !fir.ref<i1>
// CHECK:  }
// CHECK:  %[[VAL_26:.*]] = arith.constant 1 : index
// CHECK:  fir.store %[[VAL_13]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:  fir.do_loop %[[VAL_27:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_26]] {
// CHECK:    %[[VAL_28:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
// CHECK:    %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_14]] : index
// CHECK:    fir.store %[[VAL_29]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:    %[[VAL_30:.*]] = hlfir.designate %[[VAL_17]]#0 (%[[VAL_28]])  : (!fir.box<!fir.array<?xi1>>, index) -> !fir.ref<i1>
// CHECK:    %[[VAL_31:.*]] = fir.load %[[VAL_30]] : !fir.ref<i1>
// CHECK:    %[[VAL_32:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_27]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:    %[[VAL_33:.*]] = fir.convert %[[VAL_31]] : (i1) -> !fir.logical<4>
// CHECK:    fir.call @logical_value_to_numeric(%[[VAL_32]], %[[VAL_33]]) : (!fir.ref<i32>, !fir.logical<4>) -> ()
// CHECK:  }
// CHECK:  fir.freemem %[[VAL_15]] : !fir.heap<!fir.array<?xi1>>

func.func @test_saved_scalar_box(%arg0: !fir.box<!fir.type<sometype>>, %arg1: !fir.class<!fir.type<sometype>>) {
  hlfir.region_assign {
    hlfir.yield %arg0 : !fir.box<!fir.type<sometype>>
  } to {
    hlfir.yield %arg1 : !fir.class<!fir.type<sometype>>
  } user_defined_assign  (%arg2: !fir.box<!fir.type<sometype>>) to (%arg3: !fir.class<!fir.type<sometype>>) {
    fir.call @user_assign_box(%arg3, %arg2) : (!fir.class<!fir.type<sometype>>, !fir.box<!fir.type<sometype>>) -> ()
  }
  return
}
func.func private @user_assign_box(!fir.class<!fir.type<sometype>>, !fir.box<!fir.type<sometype>>) -> ()

// CHECK-LABEL:   func.func @test_saved_scalar_box(
// CHECK-SAME:                    %[[VAL_0:.*]]: !fir.box<!fir.type<sometype>>,
// CHECK-SAME:                    %[[VAL_1:.*]]: !fir.class<!fir.type<sometype>>) {
// CHECK:           %[[VAL_2:.*]] = hlfir.as_expr %[[VAL_0]] : (!fir.box<!fir.type<sometype>>) -> !hlfir.expr<!fir.type<sometype>>
// CHECK:           %[[VAL_3:.*]]:3 = hlfir.associate %[[VAL_2]]
// CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<!fir.type<sometype>>) -> !fir.box<!fir.type<sometype>>
// CHECK:           fir.call @user_assign_box(%[[VAL_1]], %[[VAL_4]]) : (!fir.class<!fir.type<sometype>>, !fir.box<!fir.type<sometype>>) -> ()
// CHECK:           hlfir.end_associate %[[VAL_3]]#1, %[[VAL_3]]#2 : !fir.ref<!fir.type<sometype>>, i1
// CHECK:           return
// CHECK:         }
