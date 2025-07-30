!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine bar()
  integer :: x, i
  x = 1
!ERROR: REDUCTION modifier on TASKLOOP directive must be DEFAULT
!ERROR: List item x must appear in EXCLUSIVE or INCLUSIVE clause of an enclosed SCAN directive
!$omp taskloop reduction(inscan, +:x)
  do i = 1, 100
    x = x + 1
  enddo
!$omp end taskloop
end

subroutine baz()
  integer :: x, i
  x = 1
!ERROR: REDUCTION modifier on TASKLOOP directive must be DEFAULT
!$omp taskloop reduction(task, +:x)
  do i = 1, 100
    x = x + 1
  enddo
!$omp end taskloop
end
