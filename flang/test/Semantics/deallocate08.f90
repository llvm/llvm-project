! RUN: %python %S/test_errors.py %s %flang_fc1
! Check there are no semantic errors in DEALLOCATE statements with EVENT_TYPE and LOCK_TYPE.

! CHECK-NOT: Object in DEALLOCATE statement is not deallocatable

subroutine deallocate_lock_event_type()
  use iso_fortran_env

  type oktype1
    type(event_type), pointer :: event
    type(lock_type), pointer :: lock
    type(notify_type), pointer ::notify 
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

  type oktype6
    type(notify_type), allocatable :: notify
  end type

  type oktype7
    type(notify_type) :: notify
  end type

  ! Variable with event_type or lock_type have to be coarrays
  ! see C1604 and 1608.
  type(oktype1), allocatable :: okt1[:]
  type(oktype2), allocatable :: okt2[:]
  type(oktype3), allocatable :: okt3[:]
  type(oktype5), allocatable :: okt5[:]
  class(oktype4), allocatable :: okt4[:]
  type(oktype6), allocatable :: okt6[:]
  type(oktype7), allocatable :: okt7[:]
  type(event_type), allocatable :: event[:]
  type(lock_type), allocatable :: lock(:)[:]
  type(notify_type), allocatable :: notify[:]

  if (.false.) then
    deallocate(okt1)
    deallocate(okt2)
    deallocate(okt3)
    deallocate(okt4)
    deallocate(okt5)
    deallocate(okt6)
    deallocate(okt7)
    deallocate(lock)
    deallocate(event)
    deallocate(notify)
  endif
end subroutine
