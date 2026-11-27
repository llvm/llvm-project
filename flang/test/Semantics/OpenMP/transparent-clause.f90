!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00(x)
  integer :: x(10)
  !ERROR: Must be a scalar value, but is a rank-1 array
  !$omp task transparent(x)
  !$omp end task
end

subroutine f01
  implicit none
  integer :: i
  !ERROR: Must have INTEGER type, but is CHARACTER(KIND=1,LEN=5_8)
  !$omp taskloop transparent("hello")
  do i = 1, 10
  end do
  !$omp end taskloop
end

