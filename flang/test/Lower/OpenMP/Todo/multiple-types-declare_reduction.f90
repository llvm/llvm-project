! Test OpenMP declare reduction with integer and real types.
! This test verifies correct lowering of user-defined reductions
! to HLFIR and their use in OpenMP parallel loops.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program main
  implicit none
  integer :: i, isum
  real    :: rsum

  !$omp declare reduction(myred: integer, real : &
  !$omp  omp_out = omp_out + omp_in) initializer(omp_priv = 0)

  isum = 0
  rsum = 0.0

  !$omp parallel do reduction(myred:isum)
  do i = 1, 3
     isum = isum + i
  end do

  !$omp parallel do reduction(myred:rsum)
  do i = 1, 3
     rsum = rsum + real(i)
  end do

  print *, isum, rsum
end program main

! Verify declare reduction is created for integer
! CHECK-LABEL: omp.declare_reduction @myred : i32
! CHECK: init {
! CHECK: arith.constant 0 : i32
! CHECK: omp.yield

! Verify integer combiner uses addi
! CHECK: combiner {
! CHECK: arith.addi
! CHECK: omp.yield

! Verify reduction is used in first parallel loop (integer)
! CHECK: omp.parallel
! CHECK: omp.wsloop
! CHECK-SAME: reduction(@myred

! Verify reduction is used in second parallel loop (real)
! CHECK: omp.parallel
! CHECK: omp.wsloop
! CHECK-SAME: reduction(@myred
! CHECK: arith.addf
