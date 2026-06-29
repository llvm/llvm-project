! Test that -fdo-concurrent-clean-nested-loops lowers a plain DO loop nested in
! a DO CONCURRENT body without the secondary-induction iter_arg: the DO variable
! is recomputed from the induction variable instead.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
! RUN: bbc -emit-hlfir -fdo-concurrent-clean-nested-loops -o - %s | FileCheck %s --check-prefixes=CHECK,CLEAN

subroutine nested(a, n)
  implicit none
  integer :: n, i, j
  integer :: a(n, n)
  do concurrent (j=1:n)
    do i = 1, n
      a(i, j) = i
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPnested
! CHECK:   %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFnestedEi"}
! CHECK:   fir.do_concurrent
! CHECK:     fir.do_concurrent.loop (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}})

! The nested plain DO loop:
! DEFAULT:     %{{.*}} = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[IV:.*]] = %{{.*}}) -> (i32) {
! DEFAULT:       fir.store %[[IV]] to %[[I_DECL]]#0 : !fir.ref<i32>

! CLEAN:       fir.do_loop %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CLEAN-NOT:     iter_args
! CLEAN:         %[[IV_CVT:.*]] = fir.convert %[[IV]] : (index) -> i32
! CLEAN:         fir.store %[[IV_CVT]] to %[[I_DECL]]#0 : !fir.ref<i32>
