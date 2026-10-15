! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! LASTPRIVATE requires a definable list item; Flang applies its existing
! definability check.

subroutine protected_target_clauses(p)
  integer, pointer, protected_target :: p
  integer :: i

  !$omp parallel private(p)
  !$omp end parallel
  !$omp parallel firstprivate(p)
  !$omp end parallel

  !ERROR: Variable 'p' on the REDUCTION clause is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  !$omp parallel do reduction(+:p)
  do i = 1, 3
  end do
  !$omp end parallel do

  !ERROR: Variable 'p' on the LASTPRIVATE clause is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  !$omp parallel do lastprivate(p)
  do i = 1, 3
  end do
  !$omp end parallel do
end subroutine
