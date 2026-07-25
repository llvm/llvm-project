! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Test while loop inside do loop.
! With the wrap-unstructured-constructs-in-execute-region pass, the inner
! `do while` is the only unstructured construct: it gets wrapped, and the
! outer counted `do` folds back to fir.do_loop.
! CHECK-LABEL: while_inside_do_loop
subroutine while_inside_do_loop
  ! CHECK-DAG: %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFwhile_inside_do_loopEi"}
  ! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ADDR]]
  ! CHECK-DAG: %[[J_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFwhile_inside_do_loopEj"}
  ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_ADDR]]
  integer :: i, j

  ! CHECK: fir.do_loop %[[I_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
  ! CHECK:   fir.store %[[I_IV]] to %[[I]]#0 : !fir.ref<i32>
  do i=8,13
    ! CHECK:   %[[C3:.*]] = arith.constant 3 : i32
    ! CHECK:   hlfir.assign %[[C3]] to %[[J]]#0 : i32, !fir.ref<i32>
    j=3

    ! CHECK:   scf.execute_region no_inline {
    ! CHECK:     cf.br ^[[HDR2:.*]]
    ! CHECK:   ^[[HDR2]]:  // 2 preds: ^{{.*}}, ^[[BODY2:.*]]
    ! CHECK:     %[[JVAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
    ! CHECK:     %[[IVAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
    ! CHECK:     %[[COND2:.*]] = arith.cmpi slt, %[[JVAL]], %[[IVAL]] : i32
    ! CHECK:     cf.cond_br %[[COND2]], ^[[BODY2]], ^[[EXIT2:.*]]
    do while (j .lt. i)
      ! CHECK:   ^[[BODY2]]:  // pred: ^[[HDR2]]
      ! CHECK:     %[[C2:.*]] = arith.constant 2 : i32
      ! CHECK:     %[[JVAL2:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
      ! CHECK:     %[[INC2:.*]] = arith.muli %[[C2]], %[[JVAL2]] : i32
      ! CHECK:     hlfir.assign %[[INC2]] to %[[J]]#0 : i32, !fir.ref<i32>
      j=j*2
    ! CHECK:     cf.br ^[[HDR2]]
    end do
    ! CHECK:   ^[[EXIT2]]:
    ! CHECK:     scf.yield
    ! CHECK:   }
  end do
  ! CHECK: }

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

    ! The outer `do while` is wrapped in scf.execute_region; the inner counted
    ! `do` lowers as fir.do_loop inside the wrap.
    ! CHECK: scf.execute_region no_inline {
    ! CHECK:   cf.br ^[[HDR1:.*]]
    ! CHECK: ^[[HDR1]]:  // 2 preds: ^{{.*}}, ^[[BODY1:.*]]
    ! CHECK:   %[[JVAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
    ! CHECK:   %[[UL:.*]] = arith.constant 21 : i32
    ! CHECK:   %[[COND:.*]] = arith.cmpi slt, %[[JVAL]], %[[UL]] : i32
    ! CHECK:   cf.cond_br %[[COND]], ^[[BODY1]], ^[[EXIT1:.*]]
    do while (j .lt. 21)
      ! CHECK: ^[[BODY1]]:  // pred: ^[[HDR1]]

      ! CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
      ! CHECK-DAG: %[[C13:.*]] = arith.constant 13 : i32
      ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
      ! CHECK: fir.do_loop %[[LI:.*]] = %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] : i32 {
        ! CHECK: fir.store %[[LI]] to %[[I]]#0 : !fir.ref<i32>
        ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
        ! CHECK: %[[J2VAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
        ! CHECK: %[[JINC:.*]] = arith.muli %[[C2]], %[[J2VAL]] : i32
        ! CHECK: hlfir.assign %[[JINC]] to %[[J]]#0 : i32, !fir.ref<i32>
      do i=8,13
        j=j*2

      ! CHECK: %[[LBIDX:.*]] = fir.convert %[[LB]] : (i32) -> index
      ! CHECK: %[[UBIDX:.*]] = fir.convert %[[UB]] : (i32) -> index
      ! CHECK: %[[STEPIDX:.*]] = fir.convert %[[STEP]] : (i32) -> index
      ! CHECK: %[[C0:.*]] = arith.constant 0 : index
      ! CHECK: %[[DIFF:.*]] = arith.subi %[[UBIDX]], %[[LBIDX]] : index
      ! CHECK: %[[ADD:.*]] = arith.addi %[[DIFF]], %[[STEPIDX]] : index
      ! CHECK: %[[TRIP:.*]] = arith.divsi %[[ADD]], %[[STEPIDX]] : index
      ! CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[TRIP]], %[[C0]] : index
      ! CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[C0]], %[[TRIP]] : index
      ! CHECK: %[[MUL:.*]] = arith.muli %[[SEL]], %[[STEPIDX]] : index
      ! CHECK: %[[LASTIDX:.*]] = arith.addi %[[LBIDX]], %[[MUL]] : index
      ! CHECK: %[[LAST:.*]] = fir.convert %[[LASTIDX]] : (index) -> i32
      ! CHECK: fir.store %[[LAST]] to %[[I]]#0 : !fir.ref<i32>
      end do

    ! CHECK:   cf.br ^[[HDR1]]
    end do
    ! CHECK: ^[[EXIT1]]:
    ! CHECK:   scf.yield
    ! CHECK: }

  ! CHECK: %[[IPRINT:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IPRINT]])
  ! CHECK: %[[JPRINT:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[JPRINT]])
  print *, i, j
end subroutine
