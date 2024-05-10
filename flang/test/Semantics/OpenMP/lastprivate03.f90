! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.2, Sections 3.2.1 & 5.3
subroutine omp_lastprivate(init)
  integer :: init
  integer :: i, a(10)
  type my_type
    integer :: val
  end type my_type
  type(my_type) :: my_var

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a LASTPRIVATE clause
  !$omp do lastprivate(a(2))
  do i=1, 10
    a(2) = init
  end do
  !$omp end do

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a LASTPRIVATE clause
  !$omp do lastprivate(my_var%val)
  do i=1, 10
    my_var%val = init
  end do
  !$omp end do
end subroutine
