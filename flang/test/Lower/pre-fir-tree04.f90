! RUN: %flang_fc1 -fdebug-pre-fir-tree %s | FileCheck %s

! Test Pre-FIR Tree captures all the coarray related statements

! CHECK: Subroutine test_coarray
Subroutine test_coarray
  use iso_fortran_env, only: team_type, event_type, lock_type
  type(team_type) :: t
  type(event_type) :: done
  type(lock_type) :: alock
  real :: y[10,*]
  integer :: counter[*]
  logical :: is_square
  ! CHECK: <<ChangeTeamConstruct>>
  change team(t, x[5,*] => y)
    ! CHECK: AssignmentStmt
    x = x[4, 1]
  end team
  ! CHECK: <<End ChangeTeamConstruct>>
  ! CHECK: FormTeamStmt
  form team(1, t)

  ! CHECK: <<IfConstruct>>
  if (this_image() == 1) then
    ! CHECK: EventPostStmt
    event post (done)
  else
    ! CHECK: EventWaitStmt
    event wait (done)
  end if
  ! CHECK: <<End IfConstruct>>

  ! CHECK: <<CriticalConstruct>>
  critical
    ! CHECK: AssignmentStmt
    counter[1] = counter[1] + 1
  end critical
  ! CHECK: <<End CriticalConstruct>>

  ! CHECK: LockStmt
  lock(alock)
  ! CHECK: PrintStmt
  print *, "I have the lock"
  ! CHECK: UnlockStmt
  unlock(alock)

  ! CHECK: SyncAllStmt
  sync all
  ! CHECK: SyncMemoryStmt
  sync memory
  ! CHECK: SyncTeamStmt
  sync team(t)

  ! CHECK: <<IfConstruct>>
  if (this_image() == 1) then
    ! CHECK: SyncImagesStmt
    sync images(*)
  else
    ! CHECK: SyncImagesStmt
    sync images(1)
  end if
  ! CHECK: <<End IfConstruct>>

  ! CHECK: <<IfConstruct!>>
  if (y<0.) then
    ! CHECK: FailImageStmt
   fail image
  end if
  ! CHECK: <<End IfConstruct!>>
end
