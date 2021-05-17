! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The ordered clause must be present on the loop construct if any ordered
! region ever binds to a loop region arising from the loop construct.

program omp_do
  integer i, j, k

  !$omp do
  do i = 1, 10
    !ERROR: The ORDERED clause must be present on the loop construct if any ORDERED region ever binds to a loop region arising from the loop construct.
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
      !ERROR: The ORDERED clause must be present on the loop construct if any ORDERED region ever binds to a loop region arising from the loop construct.
      !$omp ordered
      print *,i
      !$omp end ordered
    end do
    !$omp end parallel do
  end do
  !$omp end do

end program omp_do
