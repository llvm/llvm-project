!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61

subroutine f00(n)
  integer :: n
  integer :: i, j
  !ERROR: Must be a constant value
  !$omp fuse depth(n)
  do i = 1, 10
    do j = 1, 10
    end do
  end do
  do i = 1, 10
    do j = 1, 10
    end do
  end do
  !$omp end fuse
end
