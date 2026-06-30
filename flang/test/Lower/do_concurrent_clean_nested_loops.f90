! Test -fdo-concurrent-clean-nested-loops: a plain DO loop nested in a DO
! CONCURRENT body is lowered without the secondary-induction iter_arg (the DO
! variable is recomputed from the induction variable), while the Fortran
! post-loop value of the DO variable is still materialized after the loop.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
! RUN: bbc -emit-hlfir -fdo-concurrent-clean-nested-loops -o - %s | FileCheck %s --check-prefixes=CHECK,CLEAN

subroutine nested(a, n)
  implicit none
  integer :: n, i, j
  integer :: a(n)
  do concurrent (i=1:n)
    do j = 1, 3
    end do
    a(i) = j
  end do
end subroutine

! CHECK-LABEL:   func.func @_QPnested
! CHECK:           %[[J_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFnestedEj"}
! CHECK:           fir.do_concurrent
! CHECK:             fir.do_concurrent.loop

! Default lowering: nested loop carries the DO variable as an iter_arg and the
! post-loop value is the loop result.
! DEFAULT:           %[[RES:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
! DEFAULT:           fir.store %[[RES]] to %[[J_DECL]]#0 : !fir.ref<i32>

! Clean lowering: nested loop has no iter_arg; the post-loop value is computed
! as lb + tripCount*step after the loop and stored to the DO variable.
! CLEAN:             fir.do_loop %{{.*}} = %[[LB:.*]] to %[[UB:.*]] step %[[ST:.*]] {
! CLEAN-NOT:           iter_args
! CLEAN:             }
! CLEAN:             %[[DIFF:.*]] = arith.subi %{{.*}}, %{{.*}} overflow<nsw> : i32
! CLEAN:             %[[ADD:.*]] = arith.addi %[[DIFF]], %{{.*}} overflow<nsw> : i32
! CLEAN:             %[[TRIP:.*]] = arith.divsi %[[ADD]], %{{.*}} : i32
! CLEAN:             %[[CMP:.*]] = arith.cmpi slt, %[[TRIP]], %{{.*}} : i32
! CLEAN:             %[[SEL:.*]] = arith.select %[[CMP]], %{{.*}}, %[[TRIP]] : i32
! CLEAN:             %[[MUL:.*]] = arith.muli %[[SEL]], %{{.*}} overflow<nsw> : i32
! CLEAN:             %[[LAST:.*]] = arith.addi %{{.*}}, %[[MUL]] overflow<nsw> : i32
! CLEAN:             fir.store %[[LAST]] to %[[J_DECL]]#0 : !fir.ref<i32>
