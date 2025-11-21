! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine knownShape(array)
  integer :: array(10)

  !$omp task depend(in: array(2:8))
  !$omp end task
end subroutine

! CHECK-LABEL:   func.func @_QPknownshape(
! CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "array"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFknownshapeEarray"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 8 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant 7 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_5]]:%[[VAL_6]]:%[[VAL_7]])  shape %[[VAL_9]] : (!fir.ref<!fir.array<10xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<7xi32>>
! CHECK:           omp.task depend(taskdependin -> %[[VAL_10]] : !fir.ref<!fir.array<7xi32>>) {
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }


subroutine assumedShape(array)
  integer :: array(:)

  !$omp task depend(in: array(2:8:2))
  !$omp end task
end subroutine

! CHECK-LABEL:   func.func @_QPassumedshape(
! CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFassumedshapeEarray"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_4:.*]] = arith.constant 8 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_3]]:%[[VAL_4]]:%[[VAL_5]])  shape %[[VAL_7]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<4xi32>>
! CHECK:           %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<4xi32>>) -> !fir.ref<!fir.array<4xi32>>
! CHECK:           omp.task depend(taskdependin -> %[[VAL_9]] : !fir.ref<!fir.array<4xi32>>) {
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine vectorSubscriptArraySection(array, indices)
  integer :: array(:)
  integer :: indices(:)

  !$omp task depend (in: array(indices))
  !$omp end task
end subroutine
! CHECK-LABEL:   func.func @_QPvectorsubscriptarraysection(
! CHECK-SAME:                                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"},
! CHECK-SAME:                                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "indices"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFvectorsubscriptarraysectionEarray"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFvectorsubscriptarraysectionEindices"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi64> {
! CHECK:           ^bb0(%[[VAL_9:.*]]: index):
! CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> i64
! CHECK:             hlfir.yield_element %[[VAL_12]] : i64
! CHECK:           }
! CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_14:.*]] = hlfir.apply %[[VAL_8]], %[[VAL_13]] : (!hlfir.expr<?xi64>, index) -> i64
! CHECK:           %[[VAL_15:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_14]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
! CHECK:           omp.task depend(taskdependin -> %[[VAL_15]] : !fir.ref<i32>) {
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           hlfir.destroy %[[VAL_8]] : !hlfir.expr<?xi64>
! CHECK:           return
! CHECK:         }
