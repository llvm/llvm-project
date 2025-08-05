!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module mod
  implicit none
contains
  subroutine routine_box(a)
    implicit none
    real(4), allocatable, optional, intent(inout) :: a(:)
    integer(4) :: i

    !$omp target teams distribute parallel do shared(a)
    do i=1,10
       a(i) = i + a(i)
    end do

  end subroutine routine_box
  subroutine routine_boxchar(a)
    character(len=*), optional, intent(in) :: a
    character(len=4) :: b
    !$omp target map(from: b)
    b = a
    !$omp end target
  end subroutine routine_boxchar
end module mod

! CHECK-LABEL:   func.func @_QMmodProutine_box(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "a", fir.optional}) {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable, intent_inout, optional>, uniq_name = "_QMmodFroutine_boxEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
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


! CHECK-LABEL:   func.func @_QMmodProutine_boxchar(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "a", fir.optional}) {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.boxchar<1>
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<intent_in, optional>, uniq_name = "_QMmodFroutine_boxcharEa"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.char<1,4> {bindc_name = "b", uniq_name = "_QMmodFroutine_boxcharEb"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_4]] {uniq_name = "_QMmodFroutine_boxcharEb"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_7:.*]] = omp.map.info var_ptr(%[[VAL_6]]#1 : !fir.ref<!fir.char<1,4>>, !fir.char<1,4>) map_clauses(from) capture(ByRef) -> !fir.ref<!fir.char<1,4>> {name = "b"}
! CHECK:           %[[VAL_8:.*]] = fir.is_present %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_11:.*]]:2 = fir.if %[[VAL_8]] -> (index, index) {
! CHECK:             %[[VAL_12:.*]]:2 = fir.unboxchar %[[VAL_3]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:             fir.result %[[VAL_12]]#1, %[[VAL_10]] : index, index
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_9]], %[[VAL_9]] : index, index
! CHECK:           }
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_14:.*]]#0, %[[VAL_10]] : index
! CHECK:           %[[VAL_15:.*]] = omp.map.bounds lower_bound(%[[VAL_9]] : index) upper_bound(%[[VAL_13]] : index) extent(%[[VAL_14]]#0 : index) stride(%[[VAL_14]]#1 : index) start_idx(%[[VAL_9]] : index) {stride_in_bytes = true}
! CHECK:           %[[VAL_16:.*]] = omp.map.info var_ptr(%[[VAL_3]]#1 : !fir.ref<!fir.char<1,?>>, !fir.char<1,?>) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) bounds(%[[VAL_15]]) -> !fir.ref<!fir.char<1,?>> {name = "a"}
! CHECK:           fir.store %[[ARG0]] to %[[VAL_0]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_19:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_19]]#1, %[[VAL_18]] : index
! CHECK:           %[[VAL_21:.*]] = omp.map.bounds lower_bound(%[[VAL_17]] : index) upper_bound(%[[VAL_20]] : index) extent(%[[VAL_19]]#1 : index) stride(%[[VAL_18]] : index) start_idx(%[[VAL_17]] : index) {stride_in_bytes = true}
! CHECK:           %[[VAL_22:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.boxchar<1>>, !fir.boxchar<1>) map_clauses(implicit, to) capture(ByRef) bounds(%[[VAL_21]]) -> !fir.ref<!fir.boxchar<1>> {name = ""}
! CHECK:           omp.target map_entries(%[[VAL_7]] -> %[[VAL_23:.*]], %[[VAL_16]] -> %[[VAL_24:.*]], %[[VAL_22]] -> %[[VAL_25:.*]] : !fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,?>>, !fir.ref<!fir.boxchar<1>>) {
! CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<!fir.boxchar<1>>
! CHECK:             %[[VAL_27:.*]]:2 = fir.unboxchar %[[VAL_26]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:             %[[VAL_28:.*]] = arith.constant 4 : index
! CHECK:             %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_23]] typeparams %[[VAL_28]] {uniq_name = "_QMmodFroutine_boxcharEb"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:             %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_24]] typeparams %[[VAL_27]]#1 {fortran_attrs = #fir.var_attrs<intent_in, optional>, uniq_name = "_QMmodFroutine_boxcharEa"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:             hlfir.assign %[[VAL_30]]#0 to %[[VAL_29]]#0 : !fir.boxchar<1>, !fir.ref<!fir.char<1,4>>
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
