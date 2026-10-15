! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 C866: variable definition contexts other than assignment.

subroutine controls(index, text, flag, indices, ordinary_in)
  integer, pointer, protected_target :: index
  integer, pointer, protected_target :: indices(:)
  character(20), pointer, protected_target :: text
  logical, pointer, protected_target :: flag
  integer, intent(in) :: ordinary_in
  integer, allocatable :: allocation
  integer :: status
  namelist /group/ index
  !ERROR: 'index' may not be used as a DO variable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'index' is not definable
  do index = 1, 2
  end do
  !ERROR: The target of PROTECTED_TARGET pointer 'index' is not definable
  print *, (1, index = 1, 2)
  !ERROR: Internal file variable 'text' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'text' is not definable
  write(text, *) 1
  !ERROR: IOSTAT variable 'index' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'index' is not definable
  read(*, *, iostat=index) allocation
  !ERROR: IOMSG variable 'text' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'text' is not definable
  read(*, *, iomsg=text, iostat=status) allocation
  !ERROR: EXIST variable 'flag' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'flag' is not definable
  inquire(file='file', exist=flag)
  !ERROR: IOLENGTH variable 'index' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'index' is not definable
  inquire(iolength=index) allocation
  !ERROR: IOLENGTH variable 'indices' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'indices' is not definable
  inquire(iolength=indices(1)) allocation
  inquire(iolength=status) allocation
  inquire(iolength=ordinary_in) allocation
  !ERROR: NEWUNIT variable 'index' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'index' is not definable
  open(newunit=index, status='scratch')
  !ERROR: STAT variable 'index' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'index' is not definable
  allocate(allocation, stat=index)
  !ERROR: ERRMSG variable 'text' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'text' is not definable
  deallocate(allocation, errmsg=text)
  !ERROR: NAMELIST input group must not contain undefinable item 'index'
  !BECAUSE: The target of PROTECTED_TARGET pointer 'index' is not definable
  read(*, nml=group)
end subroutine

subroutine notify_controls(nv)
  use iso_fortran_env, only: notify_type
  type(notify_type), pointer, protected_target :: nv
  integer, save :: remote[*]
  !ERROR: The target of PROTECTED_TARGET pointer 'nv' is not definable
  !ERROR: The notify-variable must be a coarray
  notify wait(nv)
  !ERROR: The target of PROTECTED_TARGET pointer 'nv' is not definable
  remote[1, notify=nv] = 1
end subroutine

subroutine image_controls(team, event, lockvar, acquired)
  use iso_fortran_env, only: team_type, event_type, lock_type
  type(team_type), pointer, protected_target :: team
  type(event_type), pointer, protected_target :: event
  type(lock_type), pointer, protected_target :: lockvar
  logical, pointer, protected_target :: acquired
  type(lock_type) :: ordinary_lock[*]
  save ordinary_lock
  !ERROR: The target of PROTECTED_TARGET pointer 'team' is not definable
  form team(1, team)
  !ERROR: The target of PROTECTED_TARGET pointer 'event' is not definable
  event post(event)
  !ERROR: The target of PROTECTED_TARGET pointer 'event' is not definable
  event wait(event)
  !ERROR: Lock variable is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'lockvar' is not definable
  lock(lockvar)
  !ERROR: Lock variable is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'lockvar' is not definable
  unlock(lockvar)
  !ERROR: The target of PROTECTED_TARGET pointer 'acquired' is not definable
  lock(ordinary_lock, acquired_lock=acquired)
end subroutine

subroutine image_selector(index)
  integer, pointer, protected_target :: index
  integer :: remote[*]
  save remote
  !ERROR: The target of PROTECTED_TARGET pointer 'index' is not definable
  remote[1, stat=index] = 1
end subroutine
