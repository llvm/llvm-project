! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.2, Sections 3.2.1 & 5.3
subroutine omp_firstprivate(init)
  integer :: init
  integer :: a(10)
  type my_type
    integer :: val
  end type my_type
  type(my_type) :: my_var

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a FIRSTPRIVATE clause
  !$omp parallel firstprivate(a(2))
    a(2) = init
  !$omp end parallel

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a FIRSTPRIVATE clause
  !$omp parallel firstprivate(my_var%val)
    my_var%val = init
  !$omp end parallel
end subroutine
