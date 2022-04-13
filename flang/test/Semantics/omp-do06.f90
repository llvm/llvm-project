! RUN: %python %S/test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The ordered clause must be present on the loop construct if any ordered
! region ever binds to a loop region arising from the loop construct.

program omp_do
  integer i, j, k

  !$omp do
  do i = 1, 10
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered
    call my_func()
    !$omp end ordered
  end do
  !$omp end do

  !$omp do ordered private(i)
  do i = 1, 10
    !$omp parallel do
    do j = 1, 10
      print *,i
      !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
      !$omp ordered
      print *,i
      !$omp end ordered
    end do
    !$omp end parallel do
  end do
  !$omp end do

end program omp_do
