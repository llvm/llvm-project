! RUN: bbc --hlfir -o - %s | FileCheck %s

! Test jumping to the body of a do loop.
subroutine sub1()
! CHECK-LABEL:  func @_QPsub1() {
  implicit none
  integer :: i
  external foo
! CHECK:    %[[C2:.*]] = arith.constant 2 : i32
! CHECK:    %[[C0:.*]] = arith.constant 0 : i32
! CHECK:    %[[C1:.*]] = arith.constant 1 : i32
! CHECK:    %[[TRIP:.*]] = fir.alloca i32
! CHECK:    %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", {{.*}}}
! CHECK:    %[[I:.*]]:2 = hlfir.declare %[[I_REF]] {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  do i = 1, 3
    if (i .eq. 2) goto 70
! CHECK:    %[[TMP1:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK:    %[[COND:.*]] = arith.cmpi eq, %[[TMP1]], %[[C2]] : i32
! CHECK:    cf.cond_br %[[COND]], ^[[BODY:.*]], ^{{.*}}

  end do

  call foo
! CHECK:  fir.call @_QPfoo()

  do i = 1, 2
! CHECK:    fir.store %[[C2]] to %[[TRIP]] : !fir.ref<i32>
! CHECK:    fir.store %[[C1]] to %[[I]]#1 : !fir.ref<i32>
! CHECK:    cf.br ^[[HEADER:.*]]
! CHECK:  ^[[HEADER]]:
! CHECK:    %[[TMP2:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[TMP3:.*]] = arith.cmpi sgt, %[[TMP2]], %[[C0]] : i32
! CHECK:    cf.cond_br %[[TMP3]], ^[[BODY]], ^[[EXIT:.*]]

    70 call foo
! CHECK:  ^[[BODY]]:
! CHECK:    fir.call @_QPfoo()
! CHECK:    %[[TMP4:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[TMP5:.*]] = arith.subi %[[TMP4]], %[[C1]] : i32
! CHECK:    fir.store %[[TMP5]] to %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[TMP6:.*]] = fir.load %[[I]]#1 : !fir.ref<i32>
! CHECK:    %[[TMP7:.*]] = arith.addi %[[TMP6]], %[[C1]] overflow<nsw> : i32
! CHECK:    fir.store %[[TMP7]] to %[[I]]#1 : !fir.ref<i32>
! CHECK:    cf.br ^[[HEADER]]
  end do
end subroutine
! CHECK: ^[[EXIT]]:
! CHECK:    return
! CHECK:  }

! Test jumping to the body of a do loop with a step expression.
subroutine sub2()
! CHECK-LABEL:  func @_QPsub2() {
  implicit none
  integer :: i, j
  external foo
! CHECK:    %[[C_7:.*]] = arith.constant -7 : i32
! CHECK:    %[[C8:.*]] = arith.constant 8 : i32
! CHECK:    %[[C2:.*]] = arith.constant 2 : i32
! CHECK:    %[[C0:.*]] = arith.constant 0 : i32
! CHECK:    %[[C3:.*]] = arith.constant 3 : i32
! CHECK:    %[[C1:.*]] = arith.constant 1 : i32
! CHECK:    %[[TRIP:.*]] = fir.alloca i32
! CHECK:    %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i", {{.*}}}
! CHECK:    %[[I:.*]]:2 = hlfir.declare %[[I_REF]] {uniq_name = "_QFsub2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    %[[J_REF:.*]] = fir.alloca i32 {bindc_name = "j", {{.*}}}
! CHECK:    %[[J:.*]]:2 = hlfir.declare %[[J_REF]] {uniq_name = "_QFsub2Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  do i = 1, 3
    if (i .eq. 2) goto 70
! CHECK:    %[[TMP1:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK:    %[[COND:.*]] = arith.cmpi eq, %[[TMP1]], %[[C2]] : i32
! CHECK:    cf.cond_br %[[COND]], ^[[BODY:.*]], ^{{.*}}

  end do

  call foo
! CHECK:    fir.call @_QPfoo()

  j = 3
! CHECK:    hlfir.assign %[[C3]] to %[[J]]#0 : i32, !fir.ref<i32>

  do i = 1, 2, 3 * j - 8
! CHECK:    %[[TMP2:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
! CHECK:    %[[TMP3:.*]] = arith.muli %[[TMP2]], %[[C3]] overflow<nsw> : i32
! CHECK:    %[[STEP:.*]] = arith.subi %[[TMP3]], %[[C8]] overflow<nsw> : i32
! CHECK:    fir.store %[[STEP]] to %[[STEP_VAR:.*]] : !fir.ref<i32>
! CHECK:    %[[TMP4:.*]] = arith.addi %[[TMP3]], %[[C_7]] : i32
! CHECK:    %[[TMP5:.*]] = arith.divsi %[[TMP4]], %[[STEP]] : i32
! CHECK:    fir.store %[[TMP5]] to %[[TRIP]] : !fir.ref<i32>
! CHECK:    fir.store %[[C1]] to %[[I]]#1 : !fir.ref<i32>
! CHECK:    cf.br ^[[HEADER:.*]]
! CHECK:  ^[[HEADER]]:
! CHECK:    %[[TMP6:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[TMP7:.*]] = arith.cmpi sgt, %[[TMP6]], %[[C0]] : i32
! CHECK:    cf.cond_br %[[TMP7]], ^[[BODY]], ^[[EXIT:.*]]

    70 call foo
! CHECK:  ^[[BODY]]:
! CHECK:    fir.call @_QPfoo()
! CHECK:    %[[TMP8:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[TMP9:.*]] = arith.subi %[[TMP8]], %[[C1]] : i32
! CHECK:    fir.store %[[TMP9]] to %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[TMP10:.*]] = fir.load %[[I]]#1 : !fir.ref<i32>
! CHECK:    %[[STEP_VAL:.*]] = fir.load %[[STEP_VAR]] : !fir.ref<i32>
! CHECK:    %[[TMP11:.*]] = arith.addi %[[TMP10]], %[[STEP_VAL]] overflow<nsw> : i32
! CHECK:    fir.store %[[TMP11]] to %[[I]]#1 : !fir.ref<i32>
! CHECK:    cf.br ^[[HEADER]]
  end do
end subroutine
! CHECK: ^[[EXIT]]:
! CHECK:    return
! CHECK:  }
