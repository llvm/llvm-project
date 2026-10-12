! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in lock statements based on the
! statement specification in section 11.6.10 of the Fortran 2018 standard.

program test_lock_stmt
  use iso_fortran_env, only: lock_type, event_type
  implicit none

  character(len=128) error_message, msg_array(10), coindexed_msg[*], repeated_msg
  integer status, stat_array(10), coindexed_int[*], non_bool, repeated_stat
  logical non_integer, bool, bool_array(10), non_char, coindexed_logical[*], repeated_bool
  type(lock_type) :: lock_var[*], lock_array(10)[*]
  !ERROR: Variable 'non_coarray_lock' with EVENT_TYPE or LOCK_TYPE must be a coarray
  type(lock_type) :: non_coarray_lock
  type(event_type) :: not_lock_var[*]

  !___ non-standard-conforming statements ___

! type mismatches

  !ERROR: Lock variable must have type LOCK_TYPE from ISO_FORTRAN_ENV
  lock(not_lock_var)

  !ERROR: Must have LOGICAL type, but is INTEGER(4)
  lock(lock_var, acquired_lock=non_bool)

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  lock(lock_var, stat=non_integer)

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  lock(lock_var, errmsg=non_char)

! rank mismatches

  !ERROR: Must be a scalar value, but is a rank-1 array
  lock(lock_array)

  !ERROR: Must be a scalar value, but is a rank-1 array
  lock(lock_var, acquired_lock=bool_array)

  !ERROR: Must be a scalar value, but is a rank-1 array
  lock(lock_var, stat=stat_array)

  !ERROR: Must be a scalar value, but is a rank-1 array
  lock(lock_var, errmsg=msg_array)

! corank mismatch

  lock(non_coarray_lock) ! caught above

! C1173 - stat-variable and errmsg-variable shall not be a coindexed object

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  lock(lock_var, stat=coindexed_int[1])

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  lock(lock_var, errmsg=coindexed_msg[1])

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  lock(lock_var, acquired_lock=coindexed_logical[1], stat=coindexed_int[1], errmsg=coindexed_msg[1])

! C1181 - No specifier shall appear more than once in a given lock-stat-list

  !ERROR: Multiple ACQUIRED_LOCK specifiers
  lock(lock_var, acquired_lock=bool, acquired_lock=repeated_bool)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  lock(lock_var, stat=status, stat=repeated_stat)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  lock(lock_var, errmsg=error_message, errmsg=repeated_msg)

  !ERROR: Multiple ACQUIRED_LOCK specifiers
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, acquired_lock=repeated_bool)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, stat=repeated_stat)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, errmsg=repeated_msg)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  !ERROR: Multiple ACQUIRED_LOCK specifiers
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, acquired_lock=repeated_bool, stat=repeated_stat)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  !ERROR: Multiple ACQUIRED_LOCK specifiers
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, acquired_lock=repeated_bool, errmsg=repeated_msg)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, stat=repeated_stat, errmsg=repeated_msg)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  !ERROR: Multiple ACQUIRED_LOCK specifiers
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, acquired_lock=repeated_bool, stat=repeated_stat, errmsg=repeated_msg)

 contains
  subroutine lockit(x)
    type(lock_type), intent(in) :: x[*]
    !ERROR: Lock variable is not definable
    !BECAUSE: 'x' is an INTENT(IN) dummy argument
    lock(x)
    !ERROR: Lock variable is not definable
    !BECAUSE: 'x' is an INTENT(IN) dummy argument
    unlock(x)
  end
  subroutine protected_target_acquired(p, h, hp, q)
    type :: flag_holder
      logical :: flag
    end type
    type :: flag_ref
      logical, pointer, protected_target :: p
      logical, pointer :: q
    end type
    logical, pointer, protected_target :: p
    type(flag_ref) :: h
    type(flag_holder), pointer, protected_target :: hp
    logical, pointer :: q
    ! F'2028 C866 and 20.6.7(14): the target of a PROTECTED_TARGET pointer
    ! may not be an ACQUIRED_LOCK variable.
    !ERROR: ACQUIRED_LOCK variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    lock(lock_var, acquired_lock=p)
    !ERROR: ACQUIRED_LOCK variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    lock(lock_var, acquired_lock=h%p)
    !ERROR: ACQUIRED_LOCK variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'hp' is not definable
    lock(lock_var, acquired_lock=hp%flag)
    associate (a => p)
      !ERROR: ACQUIRED_LOCK variable is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      lock(lock_var, acquired_lock=a)
    end associate
    lock(lock_var, acquired_lock=q)
    lock(lock_var, acquired_lock=h%q)
  end
end program test_lock_stmt
