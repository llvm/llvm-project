! RUN: bbc -emit-fir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck %s

! Test a simple while loop.
! CHECK-LABEL: simple_loop
subroutine simple_loop
  ! CHECK: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_loopEi"}
  integer :: i

  ! CHECK: %[[C5:.*]] = arith.constant 5 : i32
  ! CHECK: fir.store %[[C5]] to %[[I_REF]]
  i = 5

  ! CHECK: br ^[[BB1:.*]]
  ! CHECK: ^[[BB1]]:  // 2 preds: ^{{.*}}, ^[[BB2:.*]]
  ! CHECK-DAG: %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C1]] : i32
  ! CHECK: cond_br %[[COND]], ^[[BB2]], ^[[BB3:.*]]
  ! CHECK: ^[[BB2]]:  // pred: ^[[BB1]]
  ! CHECK-DAG: %[[I2:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: %[[INC:.*]] = arith.subi %[[I2]], %[[C2]] : i32
  ! CHECK: fir.store %[[INC]] to %[[I_REF]] : !fir.ref<i32>
  ! CHECK: br ^[[BB1]]
  do while (i .gt. 1)
    i = i - 2
  end do

  ! CHECK: ^[[BB3]]:  // pred: ^[[BB1]]
  ! CHECK: %[[I3:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[I3]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print *, i
end subroutine

! Test 2 nested while loops.
! CHECK-LABEL: while_inside_while_loop
subroutine while_inside_while_loop
  ! CHECK-DAG: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFwhile_inside_while_loopEi"}
  ! CHECK-DAG: %[[J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFwhile_inside_while_loopEj"}
  integer :: i, j

  ! CHECK: %[[C13:.*]] = arith.constant 13 : i32
  ! CHECK: fir.store %[[C13]] to %[[I_REF]]
  i = 13

  ! CHECK: br ^[[HDR1:.*]]
  ! CHECK: ^[[HDR1]]:  // 2 preds: ^{{.*}}, ^[[EXIT2:.*]]
  ! CHECK-DAG: %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
  ! CHECK: %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C8]] : i32
  ! CHECK: cond_br %[[COND]], ^[[BODY1:.*]], ^[[EXIT1:.*]]
  do while (i .gt. 8)
    ! CHECK: ^[[BODY1]]:  // pred: ^[[HDR1]]
    ! CHECK-DAG: %[[I2:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
    ! CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
    ! CHECK: %[[INC:.*]] = arith.subi %[[I2]], %[[C5]] : i32
    ! CHECK: fir.store %[[INC]] to %[[I_REF]] : !fir.ref<i32>
    i = i - 5

    ! CHECK: %[[C3:.*]] = arith.constant 3 : i32
    ! CHECK: fir.store %[[C3]] to %[[J_REF]]
    j = 3

    ! CHECK: br ^[[HDR2:.*]]
    ! CHECK: ^[[HDR2]]:  // 2 preds: ^[[BODY1]], ^[[BODY2:.*]]
    ! CHECK-DAG: %[[J:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
    ! CHECK-DAG: %[[I3:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
    ! CHECK: %[[COND2:.*]] = arith.cmpi slt, %[[J]], %[[I3]] : i32
    ! CHECK: cond_br %[[COND2]], ^[[BODY2]], ^[[EXIT2]]
    do while (j .lt. i)
      ! CHECK: ^[[BODY2]]:  // pred: ^[[HDR2]]
      ! CHECK-DAG: %[[J2:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
      ! CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
      ! CHECK: %[[INC2:.*]] = arith.muli %[[C2]], %[[J2]] : i32
      ! CHECK: fir.store %[[INC2]] to %[[J_REF]] : !fir.ref<i32>
      j = j * 2
    ! CHECK: br ^[[HDR2]]
    end do

    ! CHECK: ^[[EXIT2]]: // pred: ^[[HDR2]]
    ! CHECK: br ^[[HDR1]]
  end do

  ! CHECK: ^[[EXIT1]]:  // pred: ^[[HDR1]]
  ! CHECK: %[[IPRINT:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[IPRINT]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK: %[[JPRINT:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[JPRINT]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print *, i, j
end subroutine
