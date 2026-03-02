! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Test while loop inside do loop.
! CHECK-LABEL: while_inside_do_loop
subroutine while_inside_do_loop
  ! CHECK-DAG: %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFwhile_inside_do_loopEi"}
  ! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ADDR]]
  ! CHECK-DAG: %[[J_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFwhile_inside_do_loopEj"}
  ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_ADDR]]
  integer :: i, j

  ! CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
  ! CHECK-DAG: %[[C13:.*]] = arith.constant 13 : i32
  ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  ! CHECK-DAG: %[[LB:.*]] = fir.convert %[[C8]] : (i32) -> index
  ! CHECK-DAG: %[[UB:.*]] = fir.convert %[[C13]] : (i32) -> index
  ! CHECK-DAG: %[[INIT:.*]] = fir.convert %[[LB]] : (index) -> i32
  ! CHECK: %[[RES:.*]] = fir.do_loop %{{.*}} = %[[LB]] to %[[UB]] step %[[C1]] iter_args(%[[I_IV:.*]] = %[[INIT]]) -> (i32) {
  ! CHECK:   fir.store %[[I_IV]] to %[[I]]#0 : !fir.ref<i32>
  do i=8,13
    ! CHECK: %[[C3:.*]] = arith.constant 3 : i32
    ! CHECK: hlfir.assign %[[C3]] to %[[J]]#0 : i32, !fir.ref<i32>
    j=3

    ! CHECK: scf.while : () -> () {
    ! CHECK: %[[JVAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
    ! CHECK: %[[IVAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
    ! CHECK: %[[COND2:.*]] = arith.cmpi slt, %[[JVAL]], %[[IVAL]] : i32
    ! CHECK: scf.condition(%[[COND2]])
    ! CHECK: } do {
    do while (j .lt. i)
      ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
      ! CHECK: %[[JVAL2:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
      ! CHECK: %[[INC2:.*]] = arith.muli %[[C2]], %[[JVAL2]] : i32
      ! CHECK: hlfir.assign %[[INC2]] to %[[J]]#0 : i32, !fir.ref<i32>
      j=j*2
    ! CHECK: scf.yield
    end do
    ! CHECK: }

  ! CHECK: %[[STEP_I32:.*]] = fir.convert %[[C1]] : (index) -> i32
  ! CHECK: %[[I_CUR:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: %[[I_NEXT:.*]] = arith.addi %[[I_CUR]], %[[STEP_I32]] overflow<nsw> : i32
  ! CHECK: fir.result %[[I_NEXT]] : i32
  ! CHECK: fir.store %[[RES]] to %[[I]]#0 : !fir.ref<i32>
  end do

  ! CHECK: %[[IPRINT:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IPRINT]])
  ! CHECK: %[[JPRINT:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[JPRINT]])
  print *, i, j
end subroutine

! Test do loop inside while loop.
! CHECK-LABEL: do_inside_while_loop
subroutine do_inside_while_loop
  ! CHECK-DAG: %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFdo_inside_while_loopEi"}
  ! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ADDR]]
  ! CHECK-DAG: %[[J_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFdo_inside_while_loopEj"}
  ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_ADDR]]
  integer :: i, j

    ! CHECK: %[[C3:.*]] = arith.constant 3 : i32
    ! CHECK: hlfir.assign %[[C3]] to %[[J]]#0 : i32, !fir.ref<i32>
    j=3

    ! CHECK: scf.while : () -> () {
    ! CHECK: %[[JVAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
    ! CHECK: %[[UL:.*]] = arith.constant 21 : i32
    ! CHECK: %[[COND:.*]] = arith.cmpi slt, %[[JVAL]], %[[UL]] : i32
    ! CHECK: scf.condition(%[[COND]])
    ! CHECK: } do {
    do while (j .lt. 21)

      ! CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
      ! CHECK-DAG: %[[C13:.*]] = arith.constant 13 : i32
      ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
      ! CHECK-DAG: %[[LB:.*]] = fir.convert %[[C8]] : (i32) -> index
      ! CHECK-DAG: %[[UB:.*]] = fir.convert %[[C13]] : (i32) -> index
      ! CHECK-DAG: %[[INIT:.*]] = fir.convert %[[LB]] : (index) -> i32
      ! CHECK: %{{.*}} = fir.do_loop %{{.*}} = %[[LB]] to %[[UB]] step %[[C1]] iter_args(%[[I_IV:.*]] = %[[INIT]]) -> (i32) {
        ! CHECK: fir.store %[[I_IV]] to %[[I]]#0 : !fir.ref<i32>
        ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
        ! CHECK: %[[J2VAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
        ! CHECK: %[[JINC:.*]] = arith.muli %[[C2]], %[[J2VAL]] : i32
        ! CHECK: hlfir.assign %[[JINC]] to %[[J]]#0 : i32, !fir.ref<i32>
        ! CHECK: %[[I_IVLOAD:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
        ! CHECK: %[[I_IVINC:.*]] = arith.addi %[[I_IVLOAD]], {{.*}} overflow<nsw> : i32
        ! CHECK: fir.result %[[I_IVINC]] : i32
      do i=8,13
        j=j*2

      ! CHECK: fir.store %{{.*}} to %[[I]]#0 : !fir.ref<i32>
      end do

    ! CHECK: scf.yield
    end do
  ! CHECK: }

  ! CHECK: %[[IPRINT:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IPRINT]])
  ! CHECK: %[[JPRINT:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[JPRINT]])
  print *, i, j
end subroutine
