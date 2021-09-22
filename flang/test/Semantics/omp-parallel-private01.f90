!RUN: %python %S/test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.15.3.3 parallel private Clause
program omp_parallel_private
  integer :: i, j, a(10), b(10), c(10)
  integer :: k = 10
  type my_type
    integer :: array(10)
  end type my_type

  type(my_type) :: my_var

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a PRIVATE or SHARED clause
  !$omp parallel private(my_var%array)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
    my_var%array(i) = k
  end do
  !$omp end parallel
end program omp_parallel_private
