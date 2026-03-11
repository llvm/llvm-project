! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine f00
  implicit none

  integer, parameter :: n = 1024
  integer :: i, j, k, array(n, n, n)

  !The i and j are predetermined private as loop induction variables nested
  !in a teams construct.
  !$omp target teams distribute default(none) shared(array)
  do i = 1, n
    do j = 1, n
      !i and j are shared in parallel
      !$omp parallel do shared(array)
      do k = 1, n
        array(i, j, k) = i + j + k
      enddo
    enddo
  enddo
end

subroutine f01
  implicit none

  integer, parameter :: n = 1024
  integer :: i, j, k, array(n, n, n)

  !$omp target teams distribute default(none) shared(array)
  do i = 1, n
    do j = 1, n
      !$omp parallel do default(none) shared(array)
      do k = 1, n
        !ERROR: The DEFAULT(NONE) clause requires that 'i' must be listed in a data-sharing attribute clause
        !ERROR: The DEFAULT(NONE) clause requires that 'j' must be listed in a data-sharing attribute clause
        array(i, j, k) = i + j + k
      enddo
    enddo
  enddo
end

