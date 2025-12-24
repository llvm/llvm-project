! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

program allocate14
  
  integer, allocatable :: i1, i2
  character(200), allocatable :: msg1, msg2
  type t
    integer, allocatable :: i
    character(10), allocatable :: msg
  end type t
  type(t) :: tt(2)
  type(t), allocatable :: ts(:)

  allocate(i1)
  allocate(msg1)

  allocate(i2, stat=i1, errmsg=msg1)
  allocate(msg2, stat=i1, errmsg=msg1)
  deallocate(i2, stat=i1, errmsg=msg1)
  deallocate(msg2, stat=i1, errmsg=msg1)

  !ERROR: STAT variable in ALLOCATE must not be the variable being allocated
  allocate(i2, stat=i2, errmsg=msg2)
  !ERROR: ERRMSG variable in ALLOCATE must not be the variable being allocated
  allocate(msg2, stat=i2, errmsg=msg2)
  !ERROR: STAT variable in DEALLOCATE must not be the variable being deallocated
  deallocate(i2, stat=i2, errmsg=msg2)
  !ERROR: ERRMSG variable in DEALLOCATE must not be the variable being deallocated
  deallocate(msg2, stat=i2, errmsg=msg2)

  allocate(tt(1)%i)
  allocate(tt(1)%msg)

  allocate(tt(2)%i, stat=tt(1)%i, errmsg=tt(1)%msg)
  allocate(tt(2)%msg, stat=tt(1)%i, errmsg=tt(1)%msg)
  deallocate(tt(2)%i, stat=tt(1)%i, errmsg=tt(1)%msg)
  deallocate(tt(2)%msg, stat=tt(1)%i, errmsg=tt(1)%msg)

  !ERROR: STAT variable in ALLOCATE must not be the variable being allocated
  allocate(tt(2)%i, stat=tt(2)%i, errmsg=tt(2)%msg)
  !ERROR: ERRMSG variable in ALLOCATE must not be the variable being allocated
  allocate(tt(2)%msg, stat=tt(2)%i, errmsg=tt(2)%msg)
  !ERROR: STAT variable in DEALLOCATE must not be the variable being deallocated
  deallocate(tt(2)%i, stat=tt(2)%i, errmsg=tt(2)%msg)
  !ERROR: ERRMSG variable in DEALLOCATE must not be the variable being deallocated
  deallocate(tt(2)%msg, stat=tt(2)%i, errmsg=tt(2)%msg)

  !TODO: STAT variable in ALLOCATE must not be the variable being allocated
  !TODO: ERRMSG variable in ALLOCATE must not be the variable being allocated
  allocate(ts(10), stat=ts(1)%i, errmsg=ts(1)%msg)
  !TODO: STAT variable in DEALLOCATE must not be the variable being deallocated
  !TODO: ERRMSG variable in DEALLOCATE must not be the variable being deallocated
  deallocate(ts, stat=ts(1)%i, errmsg=ts(1)%msg)
end program

