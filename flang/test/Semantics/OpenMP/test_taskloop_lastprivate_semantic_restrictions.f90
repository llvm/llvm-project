!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine foo()
  integer :: x, i
  x = 1
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with TASKLOOP directive is not allowed
  !$omp taskloop lastprivate(conditional: x)
  do i = 1, 100
    x = x + 1
  enddo
  !$omp end taskloop
end
