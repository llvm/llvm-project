!RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.15.3.2 parallel shared Clause
program omp_parallel_shared
  type derived
    integer :: field(2, 3)
  end type
  integer :: field(2)
  type(derived) :: y

  ! Check that derived type fields and variables with the same name
  ! don't cause errors.
  !$omp parallel
    y%field(2, 3) = 1
    field(1) = 1
  !$omp end parallel
end program omp_parallel_shared
