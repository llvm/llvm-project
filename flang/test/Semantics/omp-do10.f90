! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The DO loop iteration variable must be of type integer.

program omp_do
  real i, j, k
  !$omp do
  !ERROR: The DO loop iteration variable must be of the type integer.
  do i = 1, 10
    !ERROR: The DO loop iteration variable must be of the type integer.
    do j = 1, 10
      print *, "it", i, j
    end do
  end do
  !$omp end do

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do collapse(3)
  !ERROR: The DO loop iteration variable must be of the type integer.
  do i = 1, 10
    !ERROR: The DO loop iteration variable must be of the type integer.
    do j = 1, 10
      print *, "it", i, j
    end do
  end do
  !$omp end do

  !$omp do collapse(2)
  !ERROR: The DO loop iteration variable must be of the type integer.
  do i = 1, 10
    !ERROR: The DO loop iteration variable must be of the type integer.
    do j = 1, 10
      print *, "it", i, j
    end do
  end do
  !$omp end do

end program omp_do
