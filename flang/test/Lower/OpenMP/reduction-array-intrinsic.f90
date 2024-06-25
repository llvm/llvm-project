! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine max_array_reduction(l, r)
  integer :: l(:), r(:)

  !$omp parallel reduction(max:l)
    l = max(l, r)
  !$omp end parallel
end subroutine

! CHECK-LABEL:   omp.declare_reduction @max_byref_box_Uxi32 : !fir.ref<!fir.box<!fir.array<?xi32>>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant -2147483648 : i32
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_5]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[VAL_8:.*]] = arith.constant true
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_6]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_10]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_12:.*]] = fir.shape_shift %[[VAL_11]]#0, %[[VAL_11]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_13:.*]] = fir.rebox %[[VAL_9]]#0(%[[VAL_12]]) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_13]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           omp.yield(%[[VAL_3]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_5]]#1 step %[[VAL_7]] unordered {
! CHECK:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = arith.maxsi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
! CHECK-LABEL:   }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QPmax_array_reduction(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "l"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFmax_array_reductionEl"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFmax_array_reductionEr"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_3]]#1 to %[[VAL_5]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           omp.parallel reduction(byref @max_byref_box_Uxi32 %[[VAL_5]] -> %[[VAL_6:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) {
! CHECK:             %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFmax_array_reductionEl"} : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> (!fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.array<?xi32>>>)
! CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:             %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_9]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:             %[[VAL_11:.*]] = fir.shape %[[VAL_10]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_12:.*]] = hlfir.elemental %[[VAL_11]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:             ^bb0(%[[VAL_13:.*]]: index):
! CHECK:               %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_14]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:               %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_17:.*]] = arith.subi %[[VAL_15]]#0, %[[VAL_16]] : index
! CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_13]], %[[VAL_17]] : index
! CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_8]] (%[[VAL_18]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
! CHECK:               %[[VAL_21:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_13]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:               %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:               %[[VAL_23:.*]] = arith.cmpi sgt, %[[VAL_20]], %[[VAL_22]] : i32
! CHECK:               %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[VAL_20]], %[[VAL_22]] : i32
! CHECK:               hlfir.yield_element %[[VAL_24]] : i32
! CHECK:             }
! CHECK:             %[[VAL_25:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:             hlfir.assign %[[VAL_12]] to %[[VAL_25]] : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:             hlfir.destroy %[[VAL_12]] : !hlfir.expr<?xi32>
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
