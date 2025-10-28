! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Check for existence of loop following a DO directive

subroutine do1
  !ERROR: A DO loop must follow the DO directive
  !$omp do
end subroutine

subroutine do2
  !ERROR: A DO loop must follow the PARALLEL DO directive
  !$omp parallel do
end subroutine

subroutine do3
  !ERROR: A DO loop must follow the SIMD directive
  !$omp simd
end subroutine

subroutine do4
  !ERROR: A DO loop must follow the DO SIMD directive
  !$omp do simd
end subroutine

subroutine do5
  !ERROR: A DO loop must follow the LOOP directive
  !$omp loop
end subroutine
