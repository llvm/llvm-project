! RUN: %python %S/test_errors.py %s %flang_fc1
! Check there are no semantic errors in DEALLOCATE statements with EVENT_TYPE and LOCK_TYPE.

! CHECK-NOT: Object in DEALLOCATE statement is not deallocatable

module not_iso_fortran_env
  type event_type
  end type
  type lock_type
  end type
end module

subroutine deallocate_lock_event_type()
  use iso_fortran_env

  type oktype1
    type(event_type), pointer :: event
    type(lock_type), pointer :: lock
  end type

  type oktype5
    type(event_type), allocatable :: event
  end type

  type oktype2
    type(event_type) :: event
  end type

  type oktype3
    type(lock_type), allocatable :: lock
  end type

  type oktype4
    type(lock_type) :: lock
  end type

  ! Variable with event_type or lock_type have to be coarrays
  ! see C1604 and 1608.
  type(oktype1), allocatable :: okt1[:]
  type(oktype2), allocatable :: okt2[:]
  type(oktype3), allocatable :: okt3[:]
  type(oktype5), allocatable :: okt5[:]
  class(oktype4), allocatable :: okt4[:]
  type(event_type), allocatable :: event[:]
  type(lock_type), allocatable :: lock(:)[:]


  deallocate(okt1)
  deallocate(okt2)
  deallocate(okt3)
  deallocate(okt4)
  deallocate(okt5)
  deallocate(lock)
  deallocate(event)
end subroutine
