! Fails until we update the pass to use the `fir.do_concurrent` op.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

program main
    implicit none

    call foo(10)

    contains
        subroutine foo(n)
            implicit none
            integer :: n
            integer :: i
            integer, dimension(n) :: a

            do concurrent(i=1:n)
                a(i) = i
            end do
        end subroutine

end program main

! CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} {uniq_name = "_QFFfooEn"}

! CHECK: fir.load

! CHECK: %[[LB:.*]] = fir.convert %{{c1_.*}} : (i32) -> index
! CHECK: %[[N_VAL:.*]] = fir.load %[[N_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[UB:.*]] = fir.convert %[[N_VAL]] : (i32) -> index
! CHECK: %[[C1:.*]] = arith.constant 1 : index

! CHECK: omp.parallel {


! Verify that we resort to using the outside value for the upper bound since it
! is not originally a constant.

! CHECK:   omp.wsloop {
! CHECK:     omp.loop_nest (%{{.*}}) : index = (%[[LB]]) to (%[[UB]]) inclusive step (%{{.*}}) {
! CHECK:       omp.yield
! CHECK:     }
! CHECK:   }
! CHECK:   omp.terminator
! CHECK: }
