! Fixes a regression uncovered by Fujitsu test 0686_0024.f90. In particular,
! verifies that a pre-determined symbol is only privatized by its defining
! evaluation (e.g. the loop for which the symbol was marked as pre-determined).

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine privatize_predetermined_when_defined_by_eval
  integer::i,ii
  integer::j

  !$omp parallel
    !$omp do lastprivate(ii)
    do i=1,10
      do ii=1,10
      enddo
    enddo

    !$omp do 
    do j=1,ii
    enddo
  !$omp end parallel
end subroutine

! Verify that nothing is privatized by the `omp.parallel` op.
! CHECK: omp.parallel {

! Verify that `i` and `ii` are privatized by the first loop.
! CHECK:   omp.wsloop private(@{{.*}}ii_private_i32 %{{.*}}#0 -> %{{.*}}, @{{.*}}i_private_i32 %2#0 -> %{{.*}} : {{.*}}) {
! CHECK:   }

! Verify that `j` is privatized by the second loop.
! CHECK:   omp.wsloop private(@{{.*}}j_private_i32 %{{.*}}#0 -> %{{.*}} : {{.*}}) {
! CHECK:   }

! CHECK: }
