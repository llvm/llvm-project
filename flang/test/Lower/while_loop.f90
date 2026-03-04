! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Test a simple while loop.
! CHECK-LABEL: simple_loop
subroutine simple_loop
  ! CHECK: %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_loopEi"}
  ! CHECK: %[[I:.*]]:2 = hlfir.declare %[[I_ADDR]]
  integer :: i

  ! CHECK: %[[C5:.*]] = arith.constant 5 : i32
  ! CHECK: hlfir.assign %[[C5]] to %[[I]]#0
  i = 5

  ! CHECK: scf.while : () -> () {
  ! CHECK: %[[IVAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[COND:.*]] = arith.cmpi sgt, %[[IVAL]], %[[C1]] : i32
  ! CHECK: scf.condition(%[[COND]])
  ! CHECK: } do {
  ! CHECK: %[[IVAL2:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: %[[INC:.*]] = arith.subi %[[IVAL2]], %[[C2]] : i32
  ! CHECK: hlfir.assign %[[INC]] to %[[I]]#0 : i32, !fir.ref<i32>
  ! CHECK: scf.yield
  ! CHECK: }
  do while (i .gt. 1)
    i = i - 2
  end do

  ! CHECK: %[[IVAL3:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IVAL3]])
  print *, i
end subroutine

! Test 2 nested while loops.
! CHECK-LABEL: while_inside_while_loop
subroutine while_inside_while_loop
  ! CHECK-DAG: %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFwhile_inside_while_loopEi"}
  ! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ADDR]]
  ! CHECK-DAG: %[[J_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFwhile_inside_while_loopEj"}
  ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_ADDR]]
  integer :: i, j

  ! CHECK: %[[C13:.*]] = arith.constant 13 : i32
  ! CHECK: hlfir.assign %[[C13]] to %[[I]]#0
  i = 13

  ! CHECK: scf.while : () -> () {
  ! CHECK: %[[IVAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: %[[C8:.*]] = arith.constant 8 : i32
  ! CHECK: %[[COND:.*]] = arith.cmpi sgt, %[[IVAL]], %[[C8]] : i32
  ! CHECK: scf.condition(%[[COND]])
  ! CHECK: } do {
  do while (i .gt. 8)
    ! CHECK: %[[IVAL2:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
    ! CHECK: %[[C5:.*]] = arith.constant 5 : i32
    ! CHECK: %[[INC:.*]] = arith.subi %[[IVAL2]], %[[C5]] : i32
    ! CHECK: hlfir.assign %[[INC]] to %[[I]]#0 : i32, !fir.ref<i32>
    i = i - 5

    ! CHECK: %[[C3:.*]] = arith.constant 3 : i32
    ! CHECK: hlfir.assign %[[C3]] to %[[J]]#0
    j = 3

    ! CHECK: scf.while : () -> () {
    ! CHECK: %[[JVAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
    ! CHECK: %[[IVAL3:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
    ! CHECK: %[[COND2:.*]] = arith.cmpi slt, %[[JVAL]], %[[IVAL3]] : i32
    ! CHECK: scf.condition(%[[COND2]])
    ! CHECK: } do {
    do while (j .lt. i)
      ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
      ! CHECK: %[[JVAL2:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
      ! CHECK: %[[INC2:.*]] = arith.muli %[[C2]], %[[JVAL2]] : i32
      ! CHECK: hlfir.assign %[[INC2]] to %[[J]]#0 : i32, !fir.ref<i32>
      j = j * 2
    ! CHECK: scf.yield
    end do
    ! CHECK: }

    ! CHECK: scf.yield
  end do
  ! CHECK: }

  ! CHECK: %[[IPRINT:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IPRINT]])
  ! CHECK: %[[JPRINT:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[JPRINT]])
  print *, i, j
end subroutine
