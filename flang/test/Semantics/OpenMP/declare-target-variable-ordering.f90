! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s 2>&1 | FileCheck %s

! Test that DECLARE TARGET can appear before variable declarations in the
! specification part without causing spurious errors. This tests the fix for
! declaration ordering: within a specification part, the order of statements
! is flexible in Fortran, so !$omp declare target(x) can legally appear before
! the declaration of x as a variable.

module device_data
  implicit none
  ! DECLARE TARGET appears BEFORE the variable declaration - this is valid
  !$omp declare target(shared_buf)
  integer :: shared_buf(1000)

  ! Also test multiple names and mixed ordering
  !$omp declare target(var_a, var_b)
  real :: var_a
  integer :: var_b(10)

  ! Test that actual external procedures still work
  ! Need explicit EXTERNAL since module has implicit none
  external :: ext_proc
  !$omp declare target(ext_proc)
end module

program test_ordering
  use device_data
  implicit none

  ! Test at program scope too
  !$omp declare target(local_array)
  integer :: local_array(100)

  !$omp target map(tofrom: shared_buf, local_array)
    shared_buf(1) = 42
    local_array(1) = 24
  !$omp end target

  print *, shared_buf(1), local_array(1)
end program

subroutine ext_proc()
  !$omp declare target
  print *, "External procedure"
end subroutine

! CHECK-NOT: error:
! CHECK-NOT: already declared as a procedure
