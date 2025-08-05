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
! CHECK:           fir.store %[[VAL_3]]#0 to %[[VAL_0]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_9:.*]]:2 = fir.unboxchar %[[VAL_8]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_12:.*]]:2 = fir.unboxchar %[[VAL_8]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_12]]#1, %[[VAL_11]] : index
! CHECK:           %[[VAL_14:.*]] = omp.map.bounds lower_bound(%[[VAL_10]] : index) upper_bound(%[[VAL_13]] : index) extent(%[[VAL_12]]#1 : index) stride(%[[VAL_11]] : index) start_idx(%[[VAL_10]] : index) {stride_in_bytes = true}
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_16:.*]] = fir.box_offset %[[VAL_0]] base_addr : (!fir.ref<!fir.boxchar<1>>) -> !fir.llvm_ptr<!fir.ref<!fir.char<1,?>>>
! CHECK:           %[[VAL_17:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.boxchar<1>>, !fir.char<1,?>) map_clauses(implicit, to) capture(ByRef) var_ptr_ptr(%[[VAL_16]] : !fir.llvm_ptr<!fir.ref<!fir.char<1,?>>>) bounds(%[[VAL_14]]) -> !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_18:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.boxchar<1>>, !fir.boxchar<1>) map_clauses(to) capture(ByRef) members(%[[VAL_17]] : [0] : !fir.ref<!fir.boxchar<1>>) -> !fir.ref<!fir.boxchar<1>>
! CHECK:           omp.target map_entries(%[[VAL_7]] -> %[[VAL_19:.*]], %[[VAL_18]] -> %[[VAL_20:.*]], %[[VAL_17]] -> %[[VAL_21:.*]] : !fir.ref<!fir.char<1,4>>, !fir.ref<!fir.boxchar<1>>, !fir.ref<!fir.boxchar<1>>) private(@_QMmodFroutine_boxcharEa_firstprivate_boxchar_c8xU %[[VAL_3]]#0 -> %[[VAL_22:.*]] [map_idx=1] : !fir.boxchar<1>) {
! CHECK:             %[[VAL_23:.*]] = arith.constant 4 : index
! CHECK:             %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_19]] typeparams %[[VAL_23]] {uniq_name = "_QMmodFroutine_boxcharEb"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:             %[[VAL_25:.*]]:2 = fir.unboxchar %[[VAL_22]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:             %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_25]]#0 typeparams %[[VAL_25]]#1 {fortran_attrs = #fir.var_attrs<intent_in, optional>, uniq_name = "_QMmodFroutine_boxcharEa"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:             hlfir.assign %[[VAL_26]]#0 to %[[VAL_24]]#0 : !fir.boxchar<1>, !fir.ref<!fir.char<1,4>>
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
