! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fwrapv -o - %s | FileCheck %s --check-prefix=NO-NSW

! Tests for infinite loop.

! NO-NSW-NOT: overflow<nsw>

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
! CHECK-DAG:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK-DAG:  %[[I_DECL:.*]] = fir.declare %[[I_REF]] {{.*}}
! CHECK:  cf.br ^[[BODY1:.*]]
! CHECK: ^[[BODY1]]:
! CHECK:  %[[I:.*]] = fir.load %[[I_DECL]] : !fir.ref<i32>
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
! CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK-DAG:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : i32
! CHECK-DAG:  %[[I_DECL:.*]] = fir.declare %[[I_REF]] {{.*}}
! CHECK:  cf.br ^[[BODY1:.*]]
! CHECK: ^[[BODY1]]:
! CHECK:  %[[I:.*]] = fir.load %[[I_DECL]] : !fir.ref<i32>
! CHECK:  %[[I_NEXT:.*]] = arith.addi %[[I]], %[[C1]] : i32
! CHECK:  fir.store %[[I_NEXT]] to %[[I_DECL]] : !fir.ref<i32>
! CHECK:  %[[I:.*]] = fir.load %[[I_DECL]] : !fir.ref<i32>
! CHECK:  %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C100]] : i32
! CHECK:  cf.cond_br %[[COND]], ^[[EXIT:.*]], ^[[BODY2:.*]]
! CHECK: ^[[EXIT]]:
! CHECK:  cf.br ^[[RETURN:.*]]
! CHECK: ^[[BODY2]]:
! CHECK:  %[[I:.*]] = fir.load %[[I_DECL]] : !fir.ref<i32>
! CHECK:  %[[I_NEXT:.*]] = arith.muli %[[I]], %[[C2]] : i32
! CHECK:  fir.store %[[I_NEXT]] to %[[I_DECL]] : !fir.ref<i32>
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
! CHECK-DAG:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK-DAG:  %[[C10:.*]] = arith.constant 10 : i32
! CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[I_DECL:.*]] = fir.declare %[[I_REF]] {{.*}}
! CHECK-DAG:  %[[J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFstructured_loop_in_infiniteEj"}
! CHECK-DAG:  %[[J_DECL:.*]] = fir.declare %[[J_REF]] {{.*}}
! CHECK:  cf.br ^[[BODY1:.*]]
! CHECK: ^[[BODY1]]:
! CHECK:  %[[I:.*]] = fir.load %[[I_DECL]] : !fir.ref<i32>
! CHECK:  %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C100]] : i32
! CHECK:  cf.cond_br %[[COND]], ^[[EXIT:.*]], ^[[BODY2:.*]]
! CHECK: ^[[EXIT]]:
! CHECK:  cf.br ^[[RETURN:.*]]
! CHECK: ^[[BODY2:.*]]:
! CHECK:  fir.do_loop %[[J:[^ ]*]] =
! CHECK-SAME: %[[C1]] to %[[C10]] step %[[C1]] : i32 {
! CHECK:    fir.store %[[J]] to %[[J_DECL]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  %[[C1_INDEX:.*]] = fir.convert %[[C1]] : (i32) -> index
! CHECK:  %[[C10_INDEX:.*]] = fir.convert %[[C10]] : (i32) -> index
! CHECK:  %[[C1_STEP_INDEX:.*]] = fir.convert %[[C1]] : (i32) -> index
! CHECK:  %[[J_DIFF:.*]] = arith.subi %[[C10_INDEX]], %[[C1_INDEX]] : index
! CHECK:  %[[J_ADD:.*]] = arith.addi %[[J_DIFF]], %[[C1_STEP_INDEX]] : index
! CHECK:  %[[J_TRIP:.*]] = arith.divsi %[[J_ADD]], %[[C1_STEP_INDEX]] : index
! CHECK:  %[[J_CMP:.*]] = arith.cmpi slt, %[[J_TRIP]], %[[C0]] : index
! CHECK:  %[[J_SEL:.*]] = arith.select %[[J_CMP]], %[[C0]], %[[J_TRIP]] : index
! CHECK:  %[[J_MUL:.*]] = arith.muli %[[J_SEL]], %[[C1_STEP_INDEX]] : index
! CHECK:  %[[J_LASTIDX:.*]] = arith.addi %[[C1_INDEX]], %[[J_MUL]] : index
! CHECK:  %[[J_LAST:.*]] = fir.convert %[[J_LASTIDX]] : (index) -> i32
! CHECK:  fir.store %[[J_LAST]] to %[[J_DECL]] : !fir.ref<i32>
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
! CHECK-DAG:  %[[C50:.*]] = arith.constant 50 : i32
! CHECK-DAG:  %[[I_DECL:.*]] = fir.declare %[[I_REF]] {{.*}}
! CHECK:  cf.br ^bb1
! CHECK: ^bb1:
! CHECK:  %[[I:.*]] = fir.load %[[I_DECL]] : !fir.ref<i32>
! CHECK:  %[[COND:.*]] = arith.cmpi sgt, %[[I]], %[[C50]] : i32
! CHECK:  cf.cond_br %[[COND]], ^[[INF_HEADER:.*]], ^[[EXIT:.*]]
! CHECK: ^[[INF_HEADER]]:
! CHECK:   cf.br ^[[INF_BODY:.*]]
! CHECK: ^[[INF_BODY]]:
! CHECK:   cf.br ^[[INF_HEADER]]
! CHECK: ^[[EXIT]]:
! CHECK:  return
