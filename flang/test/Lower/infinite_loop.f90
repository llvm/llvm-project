! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -o - %s | FileCheck %s

! Tests for infinite loop.

subroutine empty_infinite()
  do
  end do
end subroutine
! CHECK-LABEL: empty_infinite
! CHECK:  cf.br ^[[BODY:.*]]
! CHECK: ^[[BODY]]:
! CHECK:  cf.br ^[[BODY]]

subroutine simple_infinite(i)
  integer :: i
  do
    if (i .gt. 100) exit
  end do
end subroutine
! CHECK-LABEL: simple_infinite
! CHECK-SAME: %[[I_REF:.*]]: !fir.ref<i32>
! CHECK:  cf.br ^[[BODY1:.*]]
! CHECK: ^[[BODY1]]:
! CHECK:  %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
! CHECK:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK:  %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C100]] : i32
! CHECK:  cf.cond_br %[[COND]], ^[[EXIT:.*]], ^[[BODY1:.*]]
! CHECK: ^[[EXIT]]:
! CHECK:  cf.br ^[[RETURN:.*]]
! CHECK: ^[[RETURN]]:
! CHECK:   return
! CHECK: }

subroutine infinite_with_two_body_blocks(i)
  integer :: i
  do
    i = i + 1
    if (i .gt. 100) exit
    i = i * 2
  end do
end subroutine
! CHECK-LABEL: infinite_with_two_body_blocks
! CHECK-SAME: %[[I_REF:.*]]: !fir.ref<i32>
! CHECK:  cf.br ^[[BODY1:.*]]
! CHECK: ^[[BODY1]]:
! CHECK:  %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[I_NEXT:.*]] = arith.addi %[[I]], %[[C1]] : i32
! CHECK:  fir.store %[[I_NEXT]] to %[[I_REF]] : !fir.ref<i32>
! CHECK:  %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
! CHECK:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK:  %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C100]] : i32
! CHECK:  cf.cond_br %[[COND]], ^[[EXIT:.*]], ^[[BODY2:.*]]
! CHECK: ^[[EXIT]]:
! CHECK:  cf.br ^[[RETURN:.*]]
! CHECK: ^[[BODY2]]:
! CHECK:  %[[C2:.*]] = arith.constant 2 : i32
! CHECK:  %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
! CHECK:  %[[I_NEXT:.*]] = arith.muli %[[C2]], %[[I]] : i32
! CHECK:  fir.store %[[I_NEXT]] to %[[I_REF]] : !fir.ref<i32>
! CHECK:  cf.br ^[[BODY1]]
! CHECK: ^[[RETURN]]:
! CHECK:   return
! CHECK: }

subroutine structured_loop_in_infinite(i)
  integer :: i
  integer :: j
  do
    if (i .gt. 100) exit
    do j=1,10
    end do
  end do
end subroutine
! CHECK-LABEL: structured_loop_in_infinite
! CHECK-SAME: %[[I_REF:.*]]: !fir.ref<i32>
! CHECK:  %[[J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFstructured_loop_in_infiniteEj"}
! CHECK:  cf.br ^[[BODY1:.*]]
! CHECK: ^[[BODY1]]:
! CHECK:  %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
! CHECK:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK:  %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C100]] : i32
! CHECK:  cf.cond_br %[[COND]], ^[[EXIT:.*]], ^[[BODY2:.*]]
! CHECK: ^[[EXIT]]:
! CHECK:  cf.br ^[[RETURN:.*]]
! CHECK: ^[[BODY2:.*]]:
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[C1_INDEX:.*]] = fir.convert %[[C1]] : (i32) -> index
! CHECK:  %[[C10:.*]] = arith.constant 10 : i32
! CHECK:  %[[C10_INDEX:.*]] = fir.convert %[[C10]] : (i32) -> index
! CHECK:  %[[C1_1:.*]] = arith.constant 1 : index
! CHECK:  %[[J_LB:.*]] = fir.convert %[[C1_INDEX]] : (index) -> i32
! CHECK:  %[[J_FINAL:.*]]:2 = fir.do_loop %[[J:[^ ]*]] =
! CHECK-SAME: %[[C1_INDEX]] to %[[C10_INDEX]] step %[[C1_1]]
! CHECK-SAME: iter_args(%[[J_IV:.*]] = %[[J_LB]]) -> (index, i32) {
! CHECK:    fir.store %[[J_IV]] to %[[J_REF]] : !fir.ref<i32>
! CHECK:    %[[J_NEXT:.*]] = arith.addi %[[J]], %[[C1_1]] : index
! CHECK:    %[[J_STEPCAST:.*]] = fir.convert %[[C1_1]] : (index) -> i32
! CHECK:    %[[J_IVLOAD:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
! CHECK:    %[[J_IVINC:.*]] = arith.addi %[[J_IVLOAD]], %[[J_STEPCAST]] : i32
! CHECK:    fir.result %[[J_NEXT]], %[[J_IVINC]] : index, i32
! CHECK:  }
! CHECK:  fir.store %[[J_FINAL]]#1 to %[[J_REF]] : !fir.ref<i32>
! CHECK:  cf.br ^[[BODY1]]
! CHECK: ^[[RETURN]]:
! CHECK:   return

subroutine empty_infinite_in_while(i)
  integer :: i
  do while (i .gt. 50)
    do
    end do
  end do
end subroutine

! CHECK-LABEL: empty_infinite_in_while
! CHECK-SAME: %[[I_REF:.*]]: !fir.ref<i32>
! CHECK:  cf.br ^bb1
! CHECK: ^bb1:
! CHECK:  %[[I:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
! CHECK:  %[[C50:.*]] = arith.constant 50 : i32
! CHECK:  %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C50]] : i32
! CHECK:  cf.cond_br %[[COND]], ^[[INF_HEADER:.*]], ^[[EXIT:.*]]
! CHECK: ^[[INF_HEADER]]:
! CHECK:   cf.br ^[[INF_BODY:.*]]
! CHECK: ^[[INF_BODY]]:
! CHECK:   cf.br ^[[INF_HEADER]]
! CHECK: ^[[EXIT]]:
! CHECK:  return
