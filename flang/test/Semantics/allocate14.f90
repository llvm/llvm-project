! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

program allocate14
  integer, allocatable :: i1, i2
  character(200), allocatable :: msg1, msg2

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
end program

