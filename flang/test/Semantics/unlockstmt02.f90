! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
program test_unlock_stmt

  use iso_fortran_env, only: lock_type

  type(LOCK_TYPE) :: locks(10)[*]
  integer :: non_lock

  !ERROR: TBD
  type(LOCK_TYPE) :: non_coarray !Invalid Declaration

  !___ non-standard-conforming statements ___

  !ERROR: TBD
  UNLOCK(non_lock)
  !ERROR: Must be a scalar value, but is a rank-1 array
  UNLOCK(locks)

 !Sync-stat-list.f90 contains invalid versions of sync-stat-list in unlock-stmt

end program test_unlock_stmt
