!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module mod
  implicit none
contains
  subroutine routine(a)
    implicit none
    real(4), allocatable, optional, intent(inout) :: a(:)
    integer(4) :: i

    !$omp target teams distribute parallel do shared(a)
        do i=1,10
            a(i) = i + a(i)
        end do

  end subroutine routine
end module mod

! CHECK-LABEL:   func.func @_QMmodProutine(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "a", fir.optional}) {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable, intent_inout, optional>, uniq_name = "_QMmodFroutineEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_8:.*]] = fir.is_present %[[VAL_2]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> i1
! CHECK:           %[[VAL_9:.*]]:5 = fir.if %[[VAL_8]] -> (index, index, index, index, index) {
! CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:             %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_14]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_12]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_18:.*]] = arith.subi %[[VAL_16]]#1, %[[VAL_11]] : index
! CHECK:             fir.result %[[VAL_17]], %[[VAL_18]], %[[VAL_16]]#1, %[[VAL_16]]#2, %[[VAL_15]]#0 : index, index, index, index, index
! CHECK:           } else {
! CHECK:             %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_20:.*]] = arith.constant -1 : index
! CHECK:             fir.result %[[VAL_19]], %[[VAL_20]], %[[VAL_19]], %[[VAL_19]], %[[VAL_19]] : index, index, index, index, index
! CHECK:           }
! CHECK:           %[[VAL_21:.*]] = omp.map.bounds lower_bound(%[[VAL_9]]#0 : index) upper_bound(%[[VAL_9]]#1 : index) extent(%[[VAL_9]]#2 : index) stride(%[[VAL_9]]#3 : index) start_idx(%[[VAL_9]]#4 : index) {stride_in_bytes = true}
! CHECK:           %[[VAL_23:.*]] = fir.is_present %[[VAL_2]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> i1
! CHECK:           fir.if %[[VAL_23]] {
! CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_2]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:             fir.store %[[VAL_24]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           }
