! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Check for existence of loop following a DO directive

subroutine do1
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp do
end subroutine

subroutine do2
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp parallel do
end subroutine

subroutine do3
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp simd
end subroutine

subroutine do4
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp do simd
end subroutine

subroutine do5
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp loop
end subroutine
