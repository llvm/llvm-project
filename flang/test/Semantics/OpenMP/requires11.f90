! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! OpenMP Version 5.0
! 2.4 Requires directive
! A unified_shared_memory requirement must be specified in every program unit
! that contains device constructs. A module and a main program in the same file
! are separate program units, each with its own specification part, so the
! REQUIRES directive in the program's specification part is not "lexically
! after" a device construct that appears in the module. This must compile
! without diagnostics.

module m
  !$omp requires unified_shared_memory
contains
  subroutine init(x, n)
    integer, intent(in) :: n
    real, intent(inout) :: x(:)
    integer :: i
    !$omp target teams distribute parallel do
    do i = 1, n
      x(i) = 1.0
    end do
  end subroutine init
end module m

program main
  use m, only: init
  !$omp requires unified_shared_memory
  integer, parameter :: n = 100
  real :: x(n)
  call init(x, n)
end program main