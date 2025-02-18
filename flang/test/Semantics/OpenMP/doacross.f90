!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine f00(x)
  integer :: x(10, 10)
  !$omp do ordered(2)
  do i = 1, 10
    do j = 1, 10
!ERROR: Duplicate variable 'i' in the iteration vector
      !$omp ordered doacross(sink: i+1, i-2)
      x(i, j) = 0
    enddo
  enddo
  !$omp end do
end

subroutine f01(x)
  integer :: x(10, 10)
  do i = 1, 10
    !$omp do ordered(1)
    do j = 1, 10
!ERROR: The iteration vector element 'i' is not an induction variable within the ORDERED loop nest
      !$omp ordered doacross(sink: i+1)
      x(i, j) = 0
    enddo
    !$omp end do
  enddo
end

