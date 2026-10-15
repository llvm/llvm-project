! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! PROTECTED_TARGET is preserved by explicit and implicit privatization.

subroutine private_pointer(p, q, w)
  integer, pointer, protected_target :: p, q
  integer, pointer :: w

  !$omp parallel private(p, w)
  p => q
  !ERROR: pointer 'w' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  w => p
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  p = 1
  !$omp end parallel
end subroutine

subroutine firstprivate_pointer(p, q, w)
  integer, pointer, protected_target :: p, q
  integer, pointer :: w

  !$omp parallel firstprivate(p) private(w)
  p => q
  !ERROR: pointer 'w' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  w => p
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  p = 1
  !$omp end parallel
end subroutine

subroutine implicit_task_pointer(p, q, w)
  integer, pointer, protected_target :: p, q
  integer, pointer :: w

  !$omp task
  p => q
  !ERROR: pointer 'w' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  w => p
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  p = 1
  !$omp end task
end subroutine
