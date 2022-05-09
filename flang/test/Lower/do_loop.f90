! RUN: bbc -emit-fir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck %s

! Simple tests for structured ordered loops with loop-control.
! Tests the structure of the loop, storage to index variable and return and 
! storage of the final value of the index variable.

! Test a simple loop with the final value of the index variable read outside the loop
! CHECK-LABEL: simple_loop
subroutine simple_loop
  ! CHECK: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_loopEi"}
  integer :: i

  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[C1_CVT:.*]] = fir.convert %c1_i32 : (i32) -> index
  ! CHECK: %[[C5:.*]] = arith.constant 5 : i32
  ! CHECK: %[[C5_CVT:.*]] = fir.convert %c5_i32 : (i32) -> index
  ! CHECK: %[[C1:.*]] = arith.constant 1 : index
  ! CHECK: %[[LI_RES:.*]] = fir.do_loop %[[LI:.*]] = %[[C1_CVT]] to %[[C5_CVT]] step %[[C1]] -> index {
  do i=1,5
  ! CHECK:   %[[LI_CVT:.*]] = fir.convert %[[LI]] : (index) -> i32
  ! CHECK:   fir.store %[[LI_CVT]] to %[[I_REF]] : !fir.ref<i32>
  ! CHECK:   %[[LI_NEXT:.*]] = arith.addi %[[LI]], %[[C1]] : index
  ! CHECK:  fir.result %[[LI_NEXT]] : index
  ! CHECK: }
  end do
  ! CHECK: %[[LI_RES_CVT:.*]] = fir.convert %[[LI_RES]] : (index) -> i32
  ! CHECK: fir.store %[[LI_RES_CVT]] to %[[I_REF]] : !fir.ref<i32>
  ! CHECK: %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK: %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[I]]) : (!fir.ref<i8>, i32) -> i1
  print *, i
end subroutine

! Test a 2-nested loop with a body composed of a reduction. Values are read from a 2d array.
! CHECK-LABEL: nested_loop
subroutine nested_loop
  ! CHECK: %[[ARR_REF:.*]] = fir.alloca !fir.array<5x5xi32> {bindc_name = "arr", uniq_name = "_QFnested_loopEarr"}
  ! CHECK: %[[ASUM_REF:.*]] = fir.alloca i32 {bindc_name = "asum", uniq_name = "_QFnested_loopEasum"}
  ! CHECK: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFnested_loopEi"}
  ! CHECK: %[[J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFnested_loopEj"}
  integer :: asum, arr(5,5)
  integer :: i, j
  asum = 0
  ! CHECK: %[[S_I:.*]] = arith.constant 1 : i32
  ! CHECK: %[[S_I_CVT:.*]] = fir.convert %[[S_I]] : (i32) -> index
  ! CHECK: %[[E_I:.*]] = arith.constant 5 : i32
  ! CHECK: %[[E_I_CVT:.*]] = fir.convert %[[E_I]] : (i32) -> index
  ! CHECK: %[[ST_I:.*]] = arith.constant 1 : index
  ! CHECK: %[[I_RES:.*]] = fir.do_loop %[[LI:.*]] = %[[S_I_CVT]] to %[[E_I_CVT]] step %[[ST_I]] -> index {
  do i=1,5
    ! CHECK: %[[LI_CVT:.*]] = fir.convert %[[LI]] : (index) -> i32
    ! CHECK: fir.store %[[LI_CVT]] to %[[I_REF]] : !fir.ref<i32>
    ! CHECK: %[[S_J:.*]] = arith.constant 1 : i32
    ! CHECK: %[[S_J_CVT:.*]] = fir.convert %[[S_J]] : (i32) -> index
    ! CHECK: %[[E_J:.*]] = arith.constant 5 : i32
    ! CHECK: %[[E_J_CVT:.*]] = fir.convert %[[E_J]] : (i32) -> index
    ! CHECK: %[[ST_J:.*]] = arith.constant 1 : index
    ! CHECK: %[[J_RES:.*]] = fir.do_loop %[[LJ:.*]] = %[[S_J_CVT]] to %[[E_J_CVT]] step %[[ST_J]] -> index {
    do j=1,5
      ! CHECK: %[[LJ_CVT:.*]] = fir.convert %[[LJ]] : (index) -> i32
      ! CHECK: fir.store %[[LJ_CVT]] to %[[J_REF]] : !fir.ref<i32>
      ! CHECK: %[[ASUM:.*]] = fir.load %[[ASUM_REF]] : !fir.ref<i32>
      ! CHECK: %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
      ! CHECK: %[[I_CVT:.*]] = fir.convert %[[I]] : (i32) -> i64
      ! CHECK: %[[C1_I:.*]] = arith.constant 1 : i64
      ! CHECK: %[[I_INDX:.*]] = arith.subi %[[I_CVT]], %[[C1_I]] : i64
      ! CHECK: %[[J:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
      ! CHECK: %[[J_CVT:.*]] = fir.convert %[[J]] : (i32) -> i64
      ! CHECK: %[[C1_J:.*]] = arith.constant 1 : i64
      ! CHECK: %[[J_INDX:.*]] = arith.subi %[[J_CVT]], %[[C1_J]] : i64
      ! CHECK: %[[ARR_IJ_REF:.*]] = fir.coordinate_of %[[ARR_REF]], %[[I_INDX]], %[[J_INDX]] : (!fir.ref<!fir.array<5x5xi32>>, i64, i64) -> !fir.ref<i32>
      ! CHECK: %[[ARR_VAL:.*]] = fir.load %[[ARR_IJ_REF]] : !fir.ref<i32>
      ! CHECK: %[[ASUM_NEW:.*]] = arith.addi %[[ASUM]], %[[ARR_VAL]] : i32
      ! CHECK: fir.store %[[ASUM_NEW]] to %[[ASUM_REF]] : !fir.ref<i32>
      asum = asum + arr(i,j)
      ! CHECK: %[[LJ_NEXT:.*]] = arith.addi %[[LJ]], %[[ST_J]] : index
      ! CHECK: fir.result %[[LJ_NEXT]] : index
    ! CHECK: }
    end do
    ! CHECK: %[[J_RES_CVT:.*]] = fir.convert %[[J_RES]] : (index) -> i32
    ! CHECK: fir.store %[[J_RES_CVT]] to %[[J_REF]] : !fir.ref<i32>
    ! CHECK: %[[LI_NEXT:.*]] = arith.addi %[[LI]], %[[ST_I]] : index
    ! CHECK: fir.result %[[LI_NEXT]] : index
  ! CHECK: }
  end do
  ! CHECK: %[[I_RES_CVT:.*]] = fir.convert %[[I_RES]] : (index) -> i32
  ! CHECK: fir.store %[[I_RES_CVT]] to %[[I_REF]] : !fir.ref<i32>
end subroutine

! Test a downcounting loop
! CHECK-LABEL: down_counting_loop
subroutine down_counting_loop()
  integer :: i
  ! CHECK: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFdown_counting_loopEi"}

  ! CHECK: %[[C5:.*]] = arith.constant 5 : i32
  ! CHECK: %[[C5_CVT:.*]] = fir.convert %[[C5]] : (i32) -> index
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[C1_CVT:.*]] = fir.convert %[[C1]] : (i32) -> index
  ! CHECK: %[[CMINUS1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[CMINUS1_STEP_CVT:.*]] = fir.convert %[[CMINUS1]] : (i32) -> index
  ! CHECK: %[[I_RES:.*]] = fir.do_loop %[[LI:.*]] = %[[C5_CVT]] to %[[C1_CVT]] step %[[CMINUS1_STEP_CVT]] -> index {
  do i=5,1,-1
  ! CHECK: %[[LI_CVT:.*]] = fir.convert %[[LI]] : (index) -> i32
  ! CHECK: fir.store %[[LI_CVT]] to %[[I_REF]] : !fir.ref<i32>
  ! CHECK: %[[LI_NEXT:.*]] = arith.addi %[[LI]], %[[CMINUS1_STEP_CVT]] : index
  ! CHECK: fir.result %[[LI_NEXT]] : index
  ! CHECK: }
  end do
  ! CHECK: %[[I_RES_CVT:.*]] = fir.convert %[[I_RES]] : (index) -> i32
  ! CHECK: fir.store %[[I_RES_CVT]] to %[[I_REF]] : !fir.ref<i32>
end subroutine

! Test a general loop with a variable step
! CHECK-LABEL: loop_with_variable_step
! CHECK-SAME: (%[[S_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}, %[[E_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "e"}, %[[ST_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "st"}) {
subroutine loop_with_variable_step(s,e,st)
  integer :: s, e, st
  ! CHECK: %[[S:.*]] = fir.load %[[S_REF]] : !fir.ref<i32>
  ! CHECK: %[[S_CVT:.*]] = fir.convert %[[S]] : (i32) -> index
  ! CHECK: %[[E:.*]] = fir.load %[[E_REF]] : !fir.ref<i32>
  ! CHECK: %[[E_CVT:.*]] = fir.convert %[[E]] : (i32) -> index
  ! CHECK: %[[ST:.*]] = fir.load %[[ST_REF]] : !fir.ref<i32>
  ! CHECK: %[[ST_CVT:.*]] = fir.convert %[[ST]] : (i32) -> index
  ! CHECK: %[[I_RES:.*]] = fir.do_loop %[[LI:.*]] = %[[S_CVT]] to %[[E_CVT]] step %[[ST_CVT]] -> index {
  do i=s,e,st
  ! CHECK:  %[[LI_CVT:.*]] = fir.convert %[[LI]] : (index) -> i32
  ! CHECK:  fir.store %[[LI_CVT]] to %[[I_REF]] : !fir.ref<i32>
  ! CHECK:  %[[LI_NEXT:.*]] = arith.addi %[[LI]], %[[ST_CVT]] : index
  ! CHECK:  fir.result %[[LI_NEXT]] : index
  ! CHECK: }
  end do
  ! CHECK: %[[I_RES_CVT:.*]] = fir.convert %[[I_RES]] : (index) -> i32
  ! CHECK: fir.store %[[I_RES_CVT]] to %[[I_REF]] : !fir.ref<i32>
end subroutine

! Test usage of pointer variables as index, start, end and step variables
! CHECK-LABEL: loop_with_pointer_variables
! CHECK-SAME: (%[[S_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "s", fir.target}, %[[E_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "e", fir.target}, %[[ST_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "st", fir.target}) {
subroutine loop_with_pointer_variables(s,e,st)
! CHECK:  %[[E_PTR_REF:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFloop_with_pointer_variablesEeptr.addr"}
! CHECK:  %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", fir.target, uniq_name = "_QFloop_with_pointer_variablesEi"}
! CHECK:  %[[I_PTR_REF:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFloop_with_pointer_variablesEiptr.addr"}
! CHECK:  %[[S_PTR_REF:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFloop_with_pointer_variablesEsptr.addr"}
! CHECK:  %[[ST_PTR_REF:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFloop_with_pointer_variablesEstptr.addr"}
  integer, target :: i
  integer, target :: s, e, st
  integer, pointer :: iptr, sptr, eptr, stptr

! CHECK:  %[[I_PTR:.*]] = fir.convert %[[I_REF]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:  fir.store %[[I_PTR]] to %[[I_PTR_REF]] : !fir.ref<!fir.ptr<i32>>
! CHECK:  %[[S_PTR:.*]] = fir.convert %[[S_REF]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:  fir.store %[[S_PTR]] to %[[S_PTR_REF]] : !fir.ref<!fir.ptr<i32>>
! CHECK:  %[[E_PTR:.*]] = fir.convert %[[E_REF]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:  fir.store %[[E_PTR]] to %[[E_PTR_REF]] : !fir.ref<!fir.ptr<i32>>
! CHECK:  %[[ST_PTR:.*]] = fir.convert %[[ST_REF]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:  fir.store %[[ST_PTR]] to %[[ST_PTR_REF]] : !fir.ref<!fir.ptr<i32>>
  iptr => i
  sptr => s
  eptr => e
  stptr => st

! CHECK:  %[[I_PTR:.*]] = fir.load %[[I_PTR_REF]] : !fir.ref<!fir.ptr<i32>>
! CHECK:  %[[S_PTR:.*]] = fir.load %[[S_PTR_REF]] : !fir.ref<!fir.ptr<i32>>
! CHECK:  %[[S:.*]] = fir.load %[[S_PTR]] : !fir.ptr<i32>
! CHECK:  %[[S_CVT:.*]] = fir.convert %[[S]] : (i32) -> index
! CHECK:  %[[E_PTR:.*]] = fir.load %[[E_PTR_REF]] : !fir.ref<!fir.ptr<i32>>
! CHECK:  %[[E:.*]] = fir.load %[[E_PTR]] : !fir.ptr<i32>
! CHECK:  %[[E_CVT:.*]] = fir.convert %[[E]] : (i32) -> index
! CHECK:  %[[ST_PTR:.*]] = fir.load %[[ST_PTR_REF]] : !fir.ref<!fir.ptr<i32>>
! CHECK:  %[[ST:.*]] = fir.load %[[ST_PTR]] : !fir.ptr<i32>
! CHECK:  %[[ST_CVT:.*]] = fir.convert %[[ST]] : (i32) -> index
! CHECK:  %[[I_RES:.*]] = fir.do_loop %[[LI:.*]] = %[[S_CVT]] to %[[E_CVT]] step %[[ST_CVT]] -> index {
  do iptr=sptr,eptr,stptr
! CHECK:    %[[LI_CVT:.*]] = fir.convert %[[LI]] : (index) -> i32
! CHECK:    fir.store %[[LI_CVT]] to %[[I_PTR]] : !fir.ptr<i32>
! CHECK:    %[[LI_NEXT:.*]] = arith.addi %[[LI]], %[[ST_CVT]] : index
! CHECK:    fir.result %[[LI_NEXT]] : index
  end do
! CHECK:  }
! CHECK:  %[[I_RES_CVT:.*]] = fir.convert %[[I_RES]] : (index) -> i32
! CHECK:  fir.store %[[I_RES_CVT:.*]] to %[[I_PTR]] : !fir.ptr<i32>
end subroutine

! Test usage of non-default integer kind for loop control and loop index variable
! CHECK-LABEL: loop_with_non_default_integer
! CHECK-SAME: (%[[S_REF:.*]]: !fir.ref<i64> {fir.bindc_name = "s"}, %[[E_REF:.*]]: !fir.ref<i64> {fir.bindc_name = "e"}, %[[ST_REF:.*]]: !fir.ref<i64> {fir.bindc_name = "st"}) {
subroutine loop_with_non_default_integer(s,e,st)
  ! CHECK: %[[I_REF:.*]] = fir.alloca i64 {bindc_name = "i", uniq_name = "_QFloop_with_non_default_integerEi"}
  integer(kind=8):: i
  ! CHECK: %[[S:.*]] = fir.load %[[S_REF]] : !fir.ref<i64>
  ! CHECK: %[[S_CVT:.*]] = fir.convert %[[S]] : (i64) -> index
  ! CHECK: %[[E:.*]] = fir.load %[[E_REF]] : !fir.ref<i64>
  ! CHECK: %[[E_CVT:.*]] = fir.convert %[[E]] : (i64) -> index
  ! CHECK: %[[ST:.*]] = fir.load %[[ST_REF]] : !fir.ref<i64>
  ! CHECK: %[[ST_CVT:.*]] = fir.convert %[[ST]] : (i64) -> index
  integer(kind=8) :: s, e, st

  ! CHECK: %[[I_RES:.*]] = fir.do_loop %[[LI:.*]] = %[[S_CVT]] to %[[E_CVT]] step %[[ST_CVT]] -> index {
  do i=s,e,st
    ! CHECK: %[[LI_CVT:.*]] = fir.convert %[[LI]] : (index) -> i64
    ! CHECK: fir.store %[[LI_CVT]] to %[[I_REF]] : !fir.ref<i64>
    ! CHECK: %[[LI_NEXT:.*]] = arith.addi %[[LI]], %[[ST_CVT]] : index
    ! CHECK: fir.result %[[LI_NEXT]] : index
  end do
  ! CHECK: }
  ! CHECK: %[[I_RES_CVT:.*]] = fir.convert %[[I_RES]] : (index) -> i64
  ! CHECK: fir.store %[[I_RES_CVT]] to %[[I_REF]] : !fir.ref<i64>
end subroutine
