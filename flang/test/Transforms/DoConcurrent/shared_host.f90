! Tests that the SHARED locality specifier works correctly through the
! do concurrent to OpenMP host conversion pipeline.
!
! SHARED variables should become implicit live-ins to omp.parallel — they
! should NOT appear on any private clause. In OpenMP, the default data sharing
! for variables in a parallel region is shared, so this is correct.
!
! A fir.local {type = shared} op should be created at the module level as
! documentation of the programmer's intent (and for future DEFAULT(NONE)
! support), but it is NOT referenced by the loop's local_vars.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

subroutine test_shared(a, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  real :: shared_val
  integer :: i

  shared_val = 42.0

  do concurrent (i = 1:n) shared(shared_val)
    a(i) = shared_val + real(i)
  end do
end subroutine

! Verify: fir.local {type = shared} exists at module level
! CHECK: fir.local {type = shared} @{{.*}}shared_val{{.*}} : f32

! Verify: the shared variable is declared and used as a live-in, NOT through
! a private clause. The omp.wsloop should have NO private clause for shared_val.
! CHECK:      %[[SHARED_VAL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("{{.*}}shared_val")
! CHECK:      omp.parallel {
! CHECK:        omp.wsloop {
! CHECK:          omp.loop_nest

! Verify: shared_val is accessed via its original SSA value (a live-in from
! outside omp.parallel), not through a block argument.
! CHECK:            fir.load %[[SHARED_VAL]]#0 : !fir.ref<f32>
