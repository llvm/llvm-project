! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Test while loop inside do loop.
! CHECK-LABEL: while_inside_do_loop
subroutine while_inside_do_loop
  ! CHECK-DAG: %[[T_REF:.*]] = fir.alloca i32
  ! CHECK-DAG: %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFwhile_inside_do_loopEi"}
  ! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ADDR]]
  ! CHECK-DAG: %[[J_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFwhile_inside_do_loopEj"}
  ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_ADDR]]
  integer :: i, j

  ! CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
  ! CHECK-DAG: %[[C13:.*]] = arith.constant 13 : i32
  ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C13]], %[[C8]] : i32
  ! CHECK: %[[RANGE:.*]] = arith.addi %[[DIFF]], %[[C1]] : i32
  ! CHECK: %[[HIGH:.*]] = arith.divsi %[[RANGE]], %[[C1]] : i32
  ! CHECK: fir.store %[[HIGH]] to %[[T_REF]] : !fir.ref<i32>
  ! CHECK: fir.store %[[C8]] to %[[I]]#0 : !fir.ref<i32>

  ! CHECK: cf.br ^[[HDR1:.*]]
  ! CHECK: ^[[HDR1]]:  // 2 preds: ^{{.*}}, ^[[EXIT2:.*]]
  ! CHECK: %[[T:.*]] = fir.load %[[T_REF]] : !fir.ref<i32>
  ! CHECK: %[[C0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[COND:.*]] = arith.cmpi sgt, %[[T]], %[[C0]] : i32
  ! CHECK: cf.cond_br %[[COND]], ^[[BODY1:.*]], ^[[EXIT1:.*]]
  do i=8,13
    ! CHECK: ^[[BODY1]]:  // pred: ^[[HDR1]]
    ! CHECK: %[[C3:.*]] = arith.constant 3 : i32
    ! CHECK: hlfir.assign %[[C3]] to %[[J]]#0 : i32, !fir.ref<i32>
    j=3

    ! CHECK: cf.br ^[[HDR2:.*]]
    ! CHECK: ^[[HDR2]]:  // 2 preds: ^[[BODY1]], ^[[BODY2:.*]]
    ! CHECK: %[[JVAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
    ! CHECK: %[[IVAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
    ! CHECK: %[[COND2:.*]] = arith.cmpi slt, %[[JVAL]], %[[IVAL]] : i32
    ! CHECK: cf.cond_br %[[COND2]], ^[[BODY2:.*]], ^[[EXIT2]]
    do while (j .lt. i)
      ! CHECK: ^[[BODY2]]:  // pred: ^[[HDR2]]
      ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
      ! CHECK: %[[JVAL2:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
      ! CHECK: %[[INC2:.*]] = arith.muli %[[C2]], %[[JVAL2]] : i32
      ! CHECK: hlfir.assign %[[INC2]] to %[[J]]#0 : i32, !fir.ref<i32>
      j=j*2
    ! CHECK: cf.br ^[[HDR2]]
    end do

  ! CHECK: ^[[EXIT2]]: // pred: ^[[HDR2]]
  ! CHECK: %[[T2:.*]] = fir.load %[[T_REF]] : !fir.ref<i32>
  ! CHECK: %[[TDEC:.*]] = arith.subi %[[T2]], {{.*}} : i32
  ! CHECK: fir.store %[[TDEC]] to %[[T_REF]]
  ! CHECK: %[[I3:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: %[[IINC:.*]] = arith.addi %[[I3]], {{.*}} overflow<nsw> : i32
  ! CHECK: fir.store %[[IINC]] to %[[I]]#0 : !fir.ref<i32>
  ! CHECK: cf.br ^[[HDR1]]
  end do

  ! CHECK: ^[[EXIT1]]:  // pred: ^[[HDR1]]
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

    ! CHECK: cf.br ^[[HDR1:.*]]
    ! CHECK: ^[[HDR1]]:  // 2 preds: ^{{.*}}, ^[[BODY1:.*]]
    ! CHECK: %[[JVAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
    ! CHECK: %[[UL:.*]] = arith.constant 21 : i32
    ! CHECK: %[[COND:.*]] = arith.cmpi slt, %[[JVAL]], %[[UL]] : i32
    ! CHECK: cf.cond_br %[[COND]], ^[[BODY1]], ^[[EXIT1:.*]]
    do while (j .lt. 21)
      ! CHECK: ^[[BODY1]]:  // pred: ^[[HDR1]]

      ! CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
      ! CHECK-DAG: %[[C13:.*]] = arith.constant 13 : i32
      ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
      ! CHECK: %{{.*}} = fir.do_loop %{{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[I_IV:.*]] = {{.*}}) -> (i32) {
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

    ! CHECK: cf.br ^[[HDR1]]
    end do

  ! CHECK: ^[[EXIT1]]:  // pred: ^[[HDR1]]
  ! CHECK: %[[IPRINT:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IPRINT]])
  ! CHECK: %[[JPRINT:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[JPRINT]])
  print *, i, j
end subroutine
