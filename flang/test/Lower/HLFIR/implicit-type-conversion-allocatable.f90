! Test implicit conversion in assignment to whole allocatables. It
! is special because care must be taken to propagate the RHS lower
! bounds to the LHS in case of re-allocation.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine preserve_lbounds(x, y)
  integer, allocatable :: x(:)
  complex, allocatable :: y(:)
  x = y
end subroutine
! CHECK-LABEL:   func.func @_QPpreserve_lbounds(
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFpreserve_lboundsEx"
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFpreserve_lboundsEy"
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:           ^bb0(%[[VAL_9:.*]]: index):
! CHECK:             %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_10]] : (!fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_13:.*]] = arith.subi %[[VAL_11]]#0, %[[VAL_12]] : index
! CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_9]], %[[VAL_13]] : index
! CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_4]] (%[[VAL_14]])  : (!fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>, index) -> !fir.ref<!fir.complex<4>>
! CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<!fir.complex<4>>
! CHECK:             %[[VAL_17:.*]] = fir.extract_value %[[VAL_16]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (f32) -> i32
! CHECK:             hlfir.yield_element %[[VAL_18]] : i32
! CHECK:           }
! CHECK:           %[[VAL_19:.*]]:3 = hlfir.associate %[[VAL_8]](%[[VAL_7]]) {uniq_name = ".tmp.keeplbounds"} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_22:.*]] = fir.shape_shift %[[VAL_21]]#0, %[[VAL_6]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_19]]#1(%[[VAL_22]]) {uniq_name = ".tmp.keeplbounds"} : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
! CHECK:           hlfir.assign %[[VAL_23]]#0 to %[[VAL_2]]#0 realloc : !fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           hlfir.end_associate %[[VAL_19]]#1, %[[VAL_19]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_8]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }
