! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fwrapv -o - %s | FileCheck %s --check-prefix=NO-NSW

! Simple tests for structured ordered loops with loop-control.
! The DO variable is recomputed from the induction variable inside the loop
! body (no secondary-induction iter_arg), and its Fortran post-loop value is
! materialized after the loop.

! NO-NSW-NOT: overflow<nsw>

! Test a simple loop with the final value of the index variable read outside the loop
! CHECK-LABEL: simple_loop
subroutine simple_loop
  ! CHECK: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_loopEi"}
  ! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]]
  integer :: i

  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[C5:.*]] = arith.constant 5 : i32
  ! CHECK: %[[C1_STEP:.*]] = arith.constant 1 : i32
  ! CHECK: fir.do_loop %[[LI:[^ ]*]] = %[[C1]] to %[[C5]] step %[[C1_STEP]] : i32 {
  do i=1,5
  ! CHECK:   fir.store %[[LI]] to %[[I_DECL]]#0 : !fir.ref<i32>
  ! CHECK: }
  end do
  ! CHECK: %[[C1_CVT:.*]] = fir.convert %[[C1]] : (i32) -> index
  ! CHECK: %[[C5_CVT:.*]] = fir.convert %[[C5]] : (i32) -> index
  ! CHECK: %[[C1_STEP_CVT:.*]] = fir.convert %[[C1_STEP]] : (i32) -> index
  ! CHECK: %[[C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C5_CVT]], %[[C1_CVT]] : index
  ! CHECK: %[[ADD:.*]] = arith.addi %[[DIFF]], %[[C1_STEP_CVT]] : index
  ! CHECK: %[[TRIP:.*]] = arith.divsi %[[ADD]], %[[C1_STEP_CVT]] : index
  ! CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[TRIP]], %[[C0]] : index
  ! CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[C0]], %[[TRIP]] : index
  ! CHECK: %[[MUL:.*]] = arith.muli %[[SEL]], %[[C1_STEP_CVT]] : index
  ! CHECK: %[[LASTIDX:.*]] = arith.addi %[[C1_CVT]], %[[MUL]] : index
  ! CHECK: %[[LAST:.*]] = fir.convert %[[LASTIDX]] : (index) -> i32
  ! CHECK: fir.store %[[LAST]] to %[[I_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[I:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[I]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print *, i
end subroutine

! Test a 2-nested loop with a body composed of a reduction. Values are read from a 2d array.
! CHECK-LABEL: nested_loop
subroutine nested_loop
  ! CHECK: %[[ARR_REF:.*]] = fir.alloca !fir.array<5x5xi32> {bindc_name = "arr", uniq_name = "_QFnested_loopEarr"}
  ! CHECK: %[[ARR_DECL:.*]]:2 = hlfir.declare %[[ARR_REF]]
  ! CHECK: %[[ASUM_REF:.*]] = fir.alloca i32 {bindc_name = "asum", uniq_name = "_QFnested_loopEasum"}
  ! CHECK: %[[ASUM_DECL:.*]]:2 = hlfir.declare %[[ASUM_REF]]
  ! CHECK: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFnested_loopEi"}
  ! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]]
  ! CHECK: %[[J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFnested_loopEj"}
  ! CHECK: %[[J_DECL:.*]]:2 = hlfir.declare %[[J_REF]]
  integer :: asum, arr(5,5)
  integer :: i, j
  asum = 0
  ! CHECK: %[[S_I:.*]] = arith.constant 1 : i32
  ! CHECK: %[[E_I:.*]] = arith.constant 5 : i32
  ! CHECK: %[[ST_I:.*]] = arith.constant 1 : i32
  ! CHECK: fir.do_loop %[[LI:[^ ]*]] = %[[S_I]] to %[[E_I]] step %[[ST_I]] : i32 {
  do i=1,5
    ! CHECK: fir.store %[[LI]] to %[[I_DECL]]#0 : !fir.ref<i32>
    ! CHECK: %[[S_J:.*]] = arith.constant 1 : i32
    ! CHECK: %[[E_J:.*]] = arith.constant 5 : i32
    ! CHECK: %[[ST_J:.*]] = arith.constant 1 : i32
    ! CHECK: fir.do_loop %[[LJ:[^ ]*]] = %[[S_J]] to %[[E_J]] step %[[ST_J]] : i32 {
    do j=1,5
      ! CHECK: fir.store %[[LJ]] to %[[J_DECL]]#0 : !fir.ref<i32>
      ! CHECK: %[[ASUM:.*]] = fir.load %[[ASUM_DECL]]#0 : !fir.ref<i32>
      ! CHECK: %[[I:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
      ! CHECK: %[[I_CVT:.*]] = fir.convert %[[I]] : (i32) -> i64
      ! CHECK: %[[J:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<i32>
      ! CHECK: %[[J_CVT:.*]] = fir.convert %[[J]] : (i32) -> i64
      ! CHECK: %[[ARR_IJ_REF:.*]] = hlfir.designate %[[ARR_DECL]]#0 (%[[I_CVT]], %[[J_CVT]])
      ! CHECK: %[[ARR_VAL:.*]] = fir.load %[[ARR_IJ_REF]] : !fir.ref<i32>
      ! CHECK: %[[ASUM_NEW:.*]] = arith.addi %[[ASUM]], %[[ARR_VAL]] : i32
      ! CHECK: hlfir.assign %[[ASUM_NEW]] to %[[ASUM_DECL]]#0 : i32, !fir.ref<i32>
      asum = asum + arr(i,j)
    ! CHECK: }
    end do
    ! CHECK: %[[S_J_CVT:.*]] = fir.convert %[[S_J]] : (i32) -> index
    ! CHECK: %[[E_J_CVT:.*]] = fir.convert %[[E_J]] : (i32) -> index
    ! CHECK: %[[ST_J_CVT:.*]] = fir.convert %[[ST_J]] : (i32) -> index
    ! CHECK: %[[J_C0:.*]] = arith.constant 0 : index
    ! CHECK: %[[J_DIFF:.*]] = arith.subi %[[E_J_CVT]], %[[S_J_CVT]] : index
    ! CHECK: %[[J_ADD:.*]] = arith.addi %[[J_DIFF]], %[[ST_J_CVT]] : index
    ! CHECK: %[[J_TRIP:.*]] = arith.divsi %[[J_ADD]], %[[ST_J_CVT]] : index
    ! CHECK: %[[J_CMP:.*]] = arith.cmpi slt, %[[J_TRIP]], %[[J_C0]] : index
    ! CHECK: %[[J_SEL:.*]] = arith.select %[[J_CMP]], %[[J_C0]], %[[J_TRIP]] : index
    ! CHECK: %[[J_MUL:.*]] = arith.muli %[[J_SEL]], %[[ST_J_CVT]] : index
    ! CHECK: %[[J_LASTIDX:.*]] = arith.addi %[[S_J_CVT]], %[[J_MUL]] : index
    ! CHECK: %[[J_LAST:.*]] = fir.convert %[[J_LASTIDX]] : (index) -> i32
    ! CHECK: fir.store %[[J_LAST]] to %[[J_DECL]]#0 : !fir.ref<i32>
  ! CHECK: }
  end do
  ! CHECK: %[[S_I_CVT:.*]] = fir.convert %[[S_I]] : (i32) -> index
  ! CHECK: %[[E_I_CVT:.*]] = fir.convert %[[E_I]] : (i32) -> index
  ! CHECK: %[[ST_I_CVT:.*]] = fir.convert %[[ST_I]] : (i32) -> index
  ! CHECK: %[[I_C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[I_DIFF:.*]] = arith.subi %[[E_I_CVT]], %[[S_I_CVT]] : index
  ! CHECK: %[[I_ADD:.*]] = arith.addi %[[I_DIFF]], %[[ST_I_CVT]] : index
  ! CHECK: %[[I_TRIP:.*]] = arith.divsi %[[I_ADD]], %[[ST_I_CVT]] : index
  ! CHECK: %[[I_CMP:.*]] = arith.cmpi slt, %[[I_TRIP]], %[[I_C0]] : index
  ! CHECK: %[[I_SEL:.*]] = arith.select %[[I_CMP]], %[[I_C0]], %[[I_TRIP]] : index
  ! CHECK: %[[I_MUL:.*]] = arith.muli %[[I_SEL]], %[[ST_I_CVT]] : index
  ! CHECK: %[[I_LASTIDX:.*]] = arith.addi %[[S_I_CVT]], %[[I_MUL]] : index
  ! CHECK: %[[I_LAST:.*]] = fir.convert %[[I_LASTIDX]] : (index) -> i32
  ! CHECK: fir.store %[[I_LAST]] to %[[I_DECL]]#0 : !fir.ref<i32>
end subroutine

! Test a downcounting loop
! CHECK-LABEL: down_counting_loop
subroutine down_counting_loop()
  integer :: i
  ! CHECK: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFdown_counting_loopEi"}
  ! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]]

  ! CHECK: %[[C5:.*]] = arith.constant 5 : i32
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[CMINUS1:.*]] = arith.constant -1 : i32
  ! CHECK: fir.do_loop %[[LI:[^ ]*]] = %[[C5]] to %[[C1]] step %[[CMINUS1]] : i32 {
  do i=5,1,-1
  ! CHECK: fir.store %[[LI]] to %[[I_DECL]]#0 : !fir.ref<i32>
  ! CHECK: }
  end do
  ! CHECK: %[[C5_CVT:.*]] = fir.convert %[[C5]] : (i32) -> index
  ! CHECK: %[[C1_CVT:.*]] = fir.convert %[[C1]] : (i32) -> index
  ! CHECK: %[[CMINUS1_STEP_CVT:.*]] = fir.convert %[[CMINUS1]] : (i32) -> index
  ! CHECK: %[[C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C1_CVT]], %[[C5_CVT]] : index
  ! CHECK: %[[ADD:.*]] = arith.addi %[[DIFF]], %[[CMINUS1_STEP_CVT]] : index
  ! CHECK: %[[TRIP:.*]] = arith.divsi %[[ADD]], %[[CMINUS1_STEP_CVT]] : index
  ! CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[TRIP]], %[[C0]] : index
  ! CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[C0]], %[[TRIP]] : index
  ! CHECK: %[[MUL:.*]] = arith.muli %[[SEL]], %[[CMINUS1_STEP_CVT]] : index
  ! CHECK: %[[LASTIDX:.*]] = arith.addi %[[C5_CVT]], %[[MUL]] : index
  ! CHECK: %[[LAST:.*]] = fir.convert %[[LASTIDX]] : (index) -> i32
  ! CHECK: fir.store %[[LAST]] to %[[I_DECL]]#0 : !fir.ref<i32>
end subroutine

! Test a general loop with a variable step
! CHECK-LABEL: loop_with_variable_step
! CHECK-SAME: (%[[S_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}, %[[E_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "e"}, %[[ST_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "st"}) {
subroutine loop_with_variable_step(s,e,st)
  integer :: s, e, st
  ! CHECK-DAG: %[[E_DECL:.*]]:2 = hlfir.declare %[[E_REF]]
  ! CHECK-DAG: %[[S_DECL:.*]]:2 = hlfir.declare %[[S_REF]]
  ! CHECK-DAG: %[[ST_DECL:.*]]:2 = hlfir.declare %[[ST_REF]]
  ! CHECK-DAG: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFloop_with_variable_stepEi"}
  ! CHECK-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]]
  ! CHECK: %[[S:.*]] = fir.load %[[S_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[E:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[ST:.*]] = fir.load %[[ST_DECL]]#0 : !fir.ref<i32>
  ! CHECK: fir.do_loop %[[LI:[^ ]*]] = %[[S]] to %[[E]] step %[[ST]] : i32 {
  do i=s,e,st
  ! CHECK:  fir.store %[[LI]] to %[[I_DECL]]#0 : !fir.ref<i32>
  ! CHECK: }
  end do
  ! CHECK: %[[S_CVT:.*]] = fir.convert %[[S]] : (i32) -> index
  ! CHECK: %[[E_CVT:.*]] = fir.convert %[[E]] : (i32) -> index
  ! CHECK: %[[ST_CVT:.*]] = fir.convert %[[ST]] : (i32) -> index
  ! CHECK: %[[C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[E_CVT]], %[[S_CVT]] : index
  ! CHECK: %[[ADD:.*]] = arith.addi %[[DIFF]], %[[ST_CVT]] : index
  ! CHECK: %[[TRIP:.*]] = arith.divsi %[[ADD]], %[[ST_CVT]] : index
  ! CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[TRIP]], %[[C0]] : index
  ! CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[C0]], %[[TRIP]] : index
  ! CHECK: %[[MUL:.*]] = arith.muli %[[SEL]], %[[ST_CVT]] : index
  ! CHECK: %[[LASTIDX:.*]] = arith.addi %[[S_CVT]], %[[MUL]] : index
  ! CHECK: %[[LAST:.*]] = fir.convert %[[LASTIDX]] : (index) -> i32
  ! CHECK: fir.store %[[LAST]] to %[[I_DECL]]#0 : !fir.ref<i32>
end subroutine

! Test usage of pointer variables as index, start, end and step variables
! CHECK-LABEL: loop_with_pointer_variables
! CHECK-SAME: (%[[S_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "s", fir.target}, %[[E_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "e", fir.target}, %[[ST_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "st", fir.target}) {
subroutine loop_with_pointer_variables(s,e,st)
! CHECK-DAG:  %[[E_PTR_REF:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "eptr", uniq_name = "_QFloop_with_pointer_variablesEeptr"}
! CHECK-DAG:  %[[E_PTR_DECL:.*]]:2 = hlfir.declare %[[E_PTR_REF]]
! CHECK-DAG:  %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", fir.target, uniq_name = "_QFloop_with_pointer_variablesEi"}
! CHECK-DAG:  %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]]
! CHECK-DAG:  %[[I_PTR_REF:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "iptr", uniq_name = "_QFloop_with_pointer_variablesEiptr"}
! CHECK-DAG:  %[[I_PTR_DECL:.*]]:2 = hlfir.declare %[[I_PTR_REF]]
! CHECK-DAG:  %[[S_PTR_REF:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "sptr", uniq_name = "_QFloop_with_pointer_variablesEsptr"}
! CHECK-DAG:  %[[S_PTR_DECL:.*]]:2 = hlfir.declare %[[S_PTR_REF]]
! CHECK-DAG:  %[[ST_PTR_REF:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "stptr", uniq_name = "_QFloop_with_pointer_variablesEstptr"}
! CHECK-DAG:  %[[ST_PTR_DECL:.*]]:2 = hlfir.declare %[[ST_PTR_REF]]
  integer, target :: i
  integer, target :: s, e, st
  integer, pointer :: iptr, sptr, eptr, stptr

! CHECK:  %[[I_PTR:.*]] = fir.embox %[[I_DECL]]#0 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:  fir.store %[[I_PTR]] to %[[I_PTR_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  iptr => i
  sptr => s
  eptr => e
  stptr => st

! CHECK:  %[[I_BOX:.*]] = fir.load %[[I_PTR_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[I_PTR:.*]] = fir.box_addr %[[I_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[S_BOX:.*]] = fir.load %[[S_PTR_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[S_PTR:.*]] = fir.box_addr %[[S_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[S:.*]] = fir.load %[[S_PTR]] : !fir.ptr<i32>
! CHECK:  %[[E_BOX:.*]] = fir.load %[[E_PTR_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[E_PTR:.*]] = fir.box_addr %[[E_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[E:.*]] = fir.load %[[E_PTR]] : !fir.ptr<i32>
! CHECK:  %[[ST_BOX:.*]] = fir.load %[[ST_PTR_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[ST_PTR:.*]] = fir.box_addr %[[ST_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[ST:.*]] = fir.load %[[ST_PTR]] : !fir.ptr<i32>
! CHECK:  fir.do_loop %[[LI:[^ ]*]] = %[[S]] to %[[E]] step %[[ST]] : i32 {
  do iptr=sptr,eptr,stptr
! CHECK:    fir.store %[[LI]] to %[[I_PTR]] : !fir.ptr<i32>
! CHECK:  }
  end do
! CHECK:  %[[S_CVT:.*]] = fir.convert %[[S]] : (i32) -> index
! CHECK:  %[[E_CVT:.*]] = fir.convert %[[E]] : (i32) -> index
! CHECK:  %[[ST_CVT:.*]] = fir.convert %[[ST]] : (i32) -> index
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[DIFF:.*]] = arith.subi %[[E_CVT]], %[[S_CVT]] : index
! CHECK:  %[[ADD:.*]] = arith.addi %[[DIFF]], %[[ST_CVT]] : index
! CHECK:  %[[TRIP:.*]] = arith.divsi %[[ADD]], %[[ST_CVT]] : index
! CHECK:  %[[CMP:.*]] = arith.cmpi slt, %[[TRIP]], %[[C0]] : index
! CHECK:  %[[SEL:.*]] = arith.select %[[CMP]], %[[C0]], %[[TRIP]] : index
! CHECK:  %[[MUL:.*]] = arith.muli %[[SEL]], %[[ST_CVT]] : index
! CHECK:  %[[LASTIDX:.*]] = arith.addi %[[S_CVT]], %[[MUL]] : index
! CHECK:  %[[LAST:.*]] = fir.convert %[[LASTIDX]] : (index) -> i32
! CHECK:  fir.store %[[LAST]] to %[[I_PTR]] : !fir.ptr<i32>
end subroutine

! Test usage of non-default integer kind for loop control and loop index variable
! CHECK-LABEL: loop_with_non_default_integer
! CHECK-SAME: (%[[S_REF:.*]]: !fir.ref<i64> {fir.bindc_name = "s"}, %[[E_REF:.*]]: !fir.ref<i64> {fir.bindc_name = "e"}, %[[ST_REF:.*]]: !fir.ref<i64> {fir.bindc_name = "st"}) {
subroutine loop_with_non_default_integer(s,e,st)
  ! CHECK-DAG: %[[E_DECL:.*]]:2 = hlfir.declare %[[E_REF]]
  ! CHECK-DAG: %[[S_DECL:.*]]:2 = hlfir.declare %[[S_REF]]
  ! CHECK-DAG: %[[ST_DECL:.*]]:2 = hlfir.declare %[[ST_REF]]
  ! CHECK-DAG: %[[I_REF:.*]] = fir.alloca i64 {bindc_name = "i", uniq_name = "_QFloop_with_non_default_integerEi"}
  ! CHECK-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]]
  integer(kind=8):: i
  ! CHECK: %[[S:.*]] = fir.load %[[S_DECL]]#0 : !fir.ref<i64>
  ! CHECK: %[[E:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i64>
  ! CHECK: %[[ST:.*]] = fir.load %[[ST_DECL]]#0 : !fir.ref<i64>
  integer(kind=8) :: s, e, st

  ! CHECK: fir.do_loop %[[LI:[^ ]*]] = %[[S]] to %[[E]] step %[[ST]] : i64 {
  do i=s,e,st
    ! CHECK: fir.store %[[LI]] to %[[I_DECL]]#0 : !fir.ref<i64>
  ! CHECK: }
  end do
  ! CHECK: %[[S_CVT:.*]] = fir.convert %[[S]] : (i64) -> index
  ! CHECK: %[[E_CVT:.*]] = fir.convert %[[E]] : (i64) -> index
  ! CHECK: %[[ST_CVT:.*]] = fir.convert %[[ST]] : (i64) -> index
  ! CHECK: %[[C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[E_CVT]], %[[S_CVT]] : index
  ! CHECK: %[[ADD:.*]] = arith.addi %[[DIFF]], %[[ST_CVT]] : index
  ! CHECK: %[[TRIP:.*]] = arith.divsi %[[ADD]], %[[ST_CVT]] : index
  ! CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[TRIP]], %[[C0]] : index
  ! CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[C0]], %[[TRIP]] : index
  ! CHECK: %[[MUL:.*]] = arith.muli %[[SEL]], %[[ST_CVT]] : index
  ! CHECK: %[[LASTIDX:.*]] = arith.addi %[[S_CVT]], %[[MUL]] : index
  ! CHECK: %[[LAST:.*]] = fir.convert %[[LASTIDX]] : (index) -> i64
  ! CHECK: fir.store %[[LAST]] to %[[I_DECL]]#0 : !fir.ref<i64>
end subroutine

! Test real loop control.
! CHECK-LABEL: loop_with_real_control
! CHECK-SAME: (%[[S_REF:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}, %[[E_REF:.*]]: !fir.ref<f32> {fir.bindc_name = "e"}, %[[ST_REF:.*]]: !fir.ref<f32> {fir.bindc_name = "st"}) {
subroutine loop_with_real_control(s,e,st)
  ! CHECK-DAG: %[[INDEX_REF:.*]] = fir.alloca index
  ! CHECK-DAG: %[[X_REF:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFloop_with_real_controlEx"}
  ! CHECK-DAG: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]]
  ! CHECK-DAG: %[[E_DECL:.*]]:2 = hlfir.declare %[[E_REF]]
  ! CHECK-DAG: %[[S_DECL:.*]]:2 = hlfir.declare %[[S_REF]]
  ! CHECK-DAG: %[[ST_DECL:.*]]:2 = hlfir.declare %[[ST_REF]]
  ! CHECK: %[[S:.*]] = fir.load %[[S_DECL]]#0 : !fir.ref<f32>
  ! CHECK: %[[E:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
  ! CHECK: %[[ST:.*]] = fir.load %[[ST_DECL]]#0 : !fir.ref<f32>
  ! CHECK: fir.store %[[ST]] to %[[ST_VAR:.*]] : !fir.ref<f32>
  real :: x, s, e, st

  ! CHECK: %[[DIFF:.*]] = arith.subf %[[E]], %[[S]] {{.*}}: f32
  ! CHECK: %[[RANGE:.*]] = arith.addf %[[DIFF]], %[[ST]] {{.*}}: f32
  ! CHECK: %[[HIGH:.*]] = arith.divf %[[RANGE]], %[[ST]] {{.*}}: f32
  ! CHECK: %[[HIGH_INDEX:.*]] = fir.convert %[[HIGH]] : (f32) -> index
  ! CHECK: fir.store %[[HIGH_INDEX]] to %[[INDEX_REF]] : !fir.ref<index>
  ! CHECK: fir.store %[[S]] to %[[X_DECL]]#0 : !fir.ref<f32>

  ! CHECK: br ^[[HDR:.*]]
  ! CHECK: ^[[HDR]]:  // 2 preds: ^{{.*}}, ^[[EXIT:.*]]
  ! CHECK-DAG: %[[INDEX:.*]] = fir.load %[[INDEX_REF]] : !fir.ref<index>
  ! CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[COND:.*]] = arith.cmpi sgt, %[[INDEX]], %[[C0]] : index
  ! CHECK: cond_br %[[COND]], ^[[BODY:.*]], ^[[EXIT:.*]]
  do x=s,e,st
    ! CHECK: ^[[BODY]]:  // pred: ^[[HDR]]
    ! CHECK-DAG: %[[INDEX2:.*]] = fir.load %[[INDEX_REF]] : !fir.ref<index>
    ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    ! CHECK: %[[INC:.*]] = arith.subi %[[INDEX2]], %[[C1]] : index
    ! CHECK: fir.store %[[INC]] to %[[INDEX_REF]] : !fir.ref<index>
    ! CHECK: %[[X2:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f32>
    ! CHECK: %[[ST_VAL:.*]] = fir.load %[[ST_VAR]] : !fir.ref<f32>
    ! CHECK: %[[XINC:.*]] = arith.addf %[[X2]], %[[ST_VAL]] {{.*}}: f32
    ! CHECK: fir.store %[[XINC]] to %[[X_DECL]]#0 : !fir.ref<f32>
    ! CHECK: br ^[[HDR]]
  end do
end subroutine
