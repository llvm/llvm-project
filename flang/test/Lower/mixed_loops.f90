! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -o - %s | FileCheck %s

! Test while loop inside do loop.
! CHECK-LABEL: while_inside_do_loop
subroutine while_inside_do_loop
  ! CHECK-DAG: %[[T_REF:.*]] = fir.alloca i32
  ! CHECK-DAG: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFwhile_inside_do_loopEi"}
  ! CHECK-DAG: %[[J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFwhile_inside_do_loopEj"}
  integer :: i, j

  ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
  ! CHECK-DAG: %[[C13:.*]] = arith.constant 13 : i32
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C13]], %[[C8]] : i32
  ! CHECK: %[[RANGE:.*]] = arith.addi %[[DIFF]], %[[C1]] : i32
  ! CHECK: %[[HIGH:.*]] = arith.divsi %[[RANGE]], %[[C1]] : i32
  ! CHECK: fir.store %[[HIGH]] to %[[T_REF]] : !fir.ref<i32>
  ! CHECK: fir.store %[[C8]] to %[[I_REF]] : !fir.ref<i32>

  ! CHECK: br ^[[HDR1:.*]]
  ! CHECK: ^[[HDR1]]:  // 2 preds: ^{{.*}}, ^[[EXIT2:.*]]
  ! CHECK-DAG: %[[T:.*]] = fir.load %[[T_REF]] : !fir.ref<i32>
  ! CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[COND:.*]] = arith.cmpi sgt, %[[T]], %[[C0]] : i32
  ! CHECK: cond_br %[[COND]], ^[[BODY1:.*]], ^[[EXIT1:.*]]
  do i=8,13
    ! CHECK: ^[[BODY1]]:  // pred: ^[[HDR1]]
    ! CHECK: %[[C3:.*]] = arith.constant 3 : i32
    ! CHECK: fir.store %[[C3]] to %[[J_REF]] : !fir.ref<i32>
    j=3

    ! CHECK: br ^[[HDR2:.*]]
    ! CHECK: ^[[HDR2]]:  // 2 preds: ^[[BODY1]], ^[[BODY2:.*]]
    ! CHECK-DAG: %[[J:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
    ! CHECK-DAG: %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
    ! CHECK: %[[COND2:.*]] = arith.cmpi slt, %[[J]], %[[I]] : i32
    ! CHECK: cond_br %[[COND2]], ^[[BODY2]], ^[[EXIT2]]
    do while (j .lt. i)
      ! CHECK: ^[[BODY2]]:  // pred: ^[[HDR2]]
      ! CHECK-DAG: %[[J2:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
      ! CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
      ! CHECK: %[[INC2:.*]] = arith.muli %[[C2]], %[[J2]] : i32
      ! CHECK: fir.store %[[INC2]] to %[[J_REF]] : !fir.ref<i32>
      j=j*2
    ! CHECK: br ^[[HDR2]]
    end do

  ! CHECK: ^[[EXIT2]]: // pred: ^[[HDR2]]
  ! CHECK-DAG: %[[T2:.*]] = fir.load %[[T_REF]] : !fir.ref<i32>
  ! CHECK-DAG: %[[C1_AGAIN:.*]] = arith.constant 1 : i32
  ! CHECK: %[[TDEC:.*]] = arith.subi %[[T2]], %[[C1_AGAIN]] : i32
  ! CHECK: fir.store %[[TDEC]] to %[[T_REF]]
  ! CHECK: %[[I3:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK: %[[C1_2:.*]] = arith.constant 1 : i32
  ! CHECK: %[[IINC:.*]] = arith.addi %[[I3]], %[[C1_2]] : i32
  ! CHECK: fir.store %[[IINC]] to %[[I_REF]] : !fir.ref<i32>
  ! CHECK: br ^[[HDR1]]
  end do

  ! CHECK: ^[[EXIT1]]:  // pred: ^[[HDR1]]
  ! CHECK: %[[IPRINT:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IPRINT]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK: %[[JPRINT:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[JPRINT]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print *, i, j
end subroutine

! Test do loop inside while loop.
! CHECK-LABEL: do_inside_while_loop
subroutine do_inside_while_loop
  ! CHECK-DAG: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFdo_inside_while_loopEi"}
  ! CHECK-DAG: %[[J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFdo_inside_while_loopEj"}
  integer :: i, j

    ! CHECK: %[[C3:.*]] = arith.constant 3 : i32
    ! CHECK: fir.store %[[C3]] to %[[J_REF]] : !fir.ref<i32>
    j=3

    ! CHECK: br ^[[HDR1:.*]]
    ! CHECK: ^[[HDR1]]:  // 2 preds: ^{{.*}}, ^[[BODY1:.*]]
    ! CHECK-DAG: %[[J:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
    ! CHECK-DAG: %[[UL:.*]] = arith.constant 21 : i32
    ! CHECK: %[[COND:.*]] = arith.cmpi slt, %[[J]], %[[UL]] : i32
    ! CHECK: cond_br %[[COND]], ^[[BODY1]], ^[[EXIT1:.*]]
    do while (j .lt. 21)
      ! CHECK: ^[[BODY1]]:  // pred: ^[[HDR1]]

      ! CHECK-DAG: %[[C8_I32:.*]] = arith.constant 8 : i32
      ! CHECK-DAG: %[[C8:.*]] = fir.convert %[[C8_I32]] : (i32) -> index
      ! CHECK-DAG: %[[C13_I32:.*]] = arith.constant 13 : i32
      ! CHECK-DAG: %[[C13:.*]] = fir.convert %[[C13_I32]] : (i32) -> index
      ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
      ! CHECK: %[[I_LB:.*]] = fir.convert %[[C8]] : (index) -> i32
      ! CHECK: %[[RESULT:.*]]:2 = fir.do_loop %[[IDX:[^ ]*]] =
      ! CHECK-SAME: %[[C8]] to %[[C13]] step %[[C1]]
      ! CHECK-SAME: iter_args(%[[I_IV:.*]] = %[[I_LB]]) -> (index, i32) {
        ! CHECK: fir.store %[[I_IV]] to %[[I_REF]] : !fir.ref<i32>
        ! CHECK-DAG: %[[J2:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
        ! CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
        ! CHECK: %[[JINC:.*]] = arith.muli %[[C2]], %[[J2]] : i32
        ! CHECK: fir.store %[[JINC]] to %[[J_REF]] : !fir.ref<i32>
        ! CHECK: %[[IINC:.*]] = arith.addi %[[IDX]], %[[C1]] : index
        ! CHECK: %[[I_STEPCAST:.*]] = fir.convert %[[C1]] : (index) -> i32
        ! CHECK: %[[I_IVLOAD:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
        ! CHECK: %[[I_IVINC:.*]] = arith.addi %[[I_IVLOAD]], %[[I_STEPCAST]] : i32
        ! CHECK: fir.result %[[IINC]], %[[I_IVINC]] : index, i32
      do i=8,13
        j=j*2

      ! CHECK: fir.store %[[RESULT]]#1 to %[[I_REF]] : !fir.ref<i32>
      end do

    ! CHECK: br ^[[HDR1]]
    end do

  ! CHECK: ^[[EXIT1]]:  // pred: ^[[HDR1]]
  ! CHECK: %[[IPRINT:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IPRINT]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK: %[[JPRINT:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[JPRINT]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print *, i, j
end subroutine
