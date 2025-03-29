! RUN: bbc %s -emit-fir -hlfir=false -o - | FileCheck %s

! Test jumping to the body of a do loop.
subroutine sub1()
! CHECK-LABEL:  func @_QPsub1() {
  implicit none
  integer :: i
  external foo
! CHECK:    %[[TRIP:.*]] = fir.alloca i32
! CHECK:    %[[I:.*]] = fir.alloca i32 {bindc_name = "i", {{.*}}}

  do i = 1, 3
    if (i .eq. 2) goto 70
! CHECK:    %[[TMP1:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:    %[[C2_1:.*]] = arith.constant 2 : i32
! CHECK:    %[[COND:.*]] = arith.cmpi eq, %[[TMP1]], %[[C2_1]] : i32
! CHECK:    cf.cond_br %[[COND]], ^[[COND_TRUE:.*]], ^{{.*}}
! CHECK:  ^[[COND_TRUE]]:
! CHECK:    cf.br ^[[BODY:.*]]

  end do

  call foo
! CHECK:  fir.call @_QPfoo()

  do i = 1, 2
! CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
! CHECK:    %[[C2_2:.*]] = arith.constant 2 : i32
! CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
! CHECK:    %[[TMP2:.*]] = arith.subi %[[C2_2]], %[[C1_1]] : i32
! CHECK:    %[[TMP3:.*]] = arith.addi %[[TMP2]], %[[C1_2]] : i32
! CHECK:    %[[TMP4:.*]] = arith.divsi %[[TMP3]], %[[C1_2]] : i32
! CHECK:    fir.store %[[TMP4]] to %[[TRIP]] : !fir.ref<i32>
! CHECK:    fir.store %[[C1_1]] to %[[I]] : !fir.ref<i32>
! CHECK:    cf.br ^[[HEADER:.*]]
! CHECK:  ^[[HEADER]]:
! CHECK:    %[[TMP5:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[C0_1:.*]] = arith.constant 0 : i32
! CHECK:    %[[TMP6:.*]] = arith.cmpi sgt, %[[TMP5]], %[[C0_1]] : i32
! CHECK:    cf.cond_br %[[TMP6]], ^[[BODY]], ^[[EXIT:.*]]

    70 call foo
! CHECK:  ^[[BODY]]:
! CHECK:    fir.call @_QPfoo()
! CHECK:    %[[TMP7:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[C1_3:.*]] = arith.constant 1 : i32
! CHECK:    %[[TMP8:.*]] = arith.subi %[[TMP7]], %[[C1_3]] : i32
! CHECK:    fir.store %[[TMP8]] to %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[TMP9:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:    %[[C1_4:.*]] = arith.constant 1 : i32
! CHECK:    %[[TMP10:.*]] = arith.addi %[[TMP9]], %[[C1_4]] overflow<nsw> : i32
! CHECK:    fir.store %[[TMP10]] to %[[I]] : !fir.ref<i32>
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
! CHECK:    %[[TRIP:.*]] = fir.alloca i32
! CHECK:    %[[I:.*]] = fir.alloca i32 {bindc_name = "i", {{.*}}}
! CHECK:    %[[J:.*]] = fir.alloca i32 {bindc_name = "j", {{.*}}}

  do i = 1, 3
    if (i .eq. 2) goto 70
! CHECK:    %[[TMP1:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:    %[[C2_1:.*]] = arith.constant 2 : i32
! CHECK:    %[[COND:.*]] = arith.cmpi eq, %[[TMP1]], %[[C2_1]] : i32
! CHECK:    cf.cond_br %[[COND]], ^[[COND_TRUE:.*]], ^{{.*}}
! CHECK:  ^[[COND_TRUE]]:
! CHECK:    cf.br ^[[BODY:.*]]

  end do

  call foo
! CHECK:    fir.call @_QPfoo()

  j = 3
! CHECK:    %[[C3_1:.*]] = arith.constant 3 : i32
! CHECK:    fir.store %[[C3_1]] to %[[J]] : !fir.ref<i32>

  do i = 1, 2, 3 * j - 8
! CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
! CHECK:    %[[C2_2:.*]] = arith.constant 2 : i32
! CHECK:    %[[C3_2:.*]] = arith.constant 3 : i32
! CHECK:    %[[TMP2:.*]] = fir.load %[[J]] : !fir.ref<i32>
! CHECK:    %[[TMP3:.*]] = arith.muli %[[C3_2]], %[[TMP2]] overflow<nsw> : i32
! CHECK:    %[[C8_1:.*]] = arith.constant 8 : i32
! CHECK:    %[[STEP:.*]] = arith.subi %[[TMP3]], %[[C8_1]] overflow<nsw> : i32
! CHECK:    fir.store %[[STEP]] to %[[STEP_VAR:.*]] : !fir.ref<i32>
! CHECK:    %[[TMP4:.*]] = arith.subi %[[C2_2]], %[[C1_1]] : i32
! CHECK:    %[[TMP5:.*]] = arith.addi %[[TMP4]], %[[STEP]] : i32
! CHECK:    %[[TMP6:.*]] = arith.divsi %[[TMP5]], %[[STEP]] : i32
! CHECK:    fir.store %[[TMP6]] to %[[TRIP]] : !fir.ref<i32>
! CHECK:    fir.store %[[C1_1]] to %[[I]] : !fir.ref<i32>
! CHECK:    cf.br ^[[HEADER:.*]]
! CHECK:  ^[[HEADER]]:
! CHECK:    %[[TMP7:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[C0_1:.*]] = arith.constant 0 : i32
! CHECK:    %[[TMP8:.*]] = arith.cmpi sgt, %[[TMP7]], %[[C0_1]] : i32
! CHECK:    cf.cond_br %[[TMP8]], ^[[BODY]], ^[[EXIT:.*]]

    70 call foo
! CHECK:  ^[[BODY]]:
! CHECK:    fir.call @_QPfoo()
! CHECK:    %[[TMP9:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
! CHECK:    %[[TMP10:.*]] = arith.subi %[[TMP9]], %[[C1_2]] : i32
! CHECK:    fir.store %[[TMP10]] to %[[TRIP]] : !fir.ref<i32>
! CHECK:    %[[TMP11:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:    %[[STEP_VAL:.*]] = fir.load %[[STEP_VAR]] : !fir.ref<i32>
! CHECK:    %[[TMP12:.*]] = arith.addi %[[TMP11]], %[[STEP_VAL]] overflow<nsw> : i32
! CHECK:    fir.store %[[TMP12]] to %[[I]] : !fir.ref<i32>
! CHECK:    cf.br ^[[HEADER]]
  end do
end subroutine
! CHECK: ^[[EXIT]]:
! CHECK:    return
! CHECK:  }
