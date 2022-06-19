! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

module not_iso_fortran_env
  type event_type
  end type
  type lock_type
  end type
end module

subroutine C948_a()
! If SOURCE= appears, the declared type of source-expr shall not be EVENT_TYPE
! or LOCK_-TYPE from the intrinsic module ISO_FORTRAN_ENV, or have a potential subobject
! component of type EVENT_TYPE or LOCK_TYPE.
  use iso_fortran_env

  type oktype1
    type(event_type), pointer :: event
    type(lock_type), pointer :: lock
  end type

  type oktype2
    class(oktype1), allocatable :: t1a
    type(oktype1) :: t1b
  end type

  type, extends(oktype1) :: oktype3
    real, allocatable :: x(:)
  end type

  type noktype1
    type(event_type), allocatable :: event
  end type

  type noktype2
    type(event_type) :: event
  end type

  type noktype3
    type(lock_type), allocatable :: lock
  end type

  type noktype4
    type(lock_type) :: lock
  end type

  type, extends(noktype4) :: noktype5
    real, allocatable :: x(:)
  end type

  type, extends(event_type) :: noktype6
    real, allocatable :: x(:)
  end type

  type recursiveType
    real x(10)
    type(recursiveType), allocatable :: next
  end type

  type recursiveTypeNok
    real x(10)
    type(recursiveType), allocatable :: next
    type(noktype5), allocatable :: trouble
  end type

  ! variable with event_type or lock_type have to be coarrays
  ! see C1604 and 1608.
  type(oktype1), allocatable :: okt1[:]
  class(oktype2), allocatable :: okt2(:)[:]
  type(oktype3), allocatable :: okt3[:]
  type(noktype1), allocatable :: nokt1[:]
  type(noktype2), allocatable :: nokt2[:]
  class(noktype3), allocatable :: nokt3[:]
  type(noktype4), allocatable :: nokt4[:]
  type(noktype5), allocatable :: nokt5[:]
  class(noktype6), allocatable :: nokt6(:)[:]
  type(event_type), allocatable :: event[:]
  type(lock_type), allocatable :: lock(:)[:]
  class(recursiveType), allocatable :: recok
  type(recursiveTypeNok), allocatable :: recnok[:]
  class(*), allocatable :: whatever[:]

  type(oktype1), allocatable :: okt1src[:]
  class(oktype2), allocatable :: okt2src(:)[:]
  type(oktype3), allocatable :: okt3src[:]
  class(noktype1), allocatable :: nokt1src[:]
  type(noktype2), allocatable :: nokt2src[:]
  type(noktype3), allocatable :: nokt3src[:]
  class(noktype4), allocatable :: nokt4src[:]
  type(noktype5), allocatable :: nokt5src[:]
  class(noktype6), allocatable :: nokt6src(:)[:]
  type(event_type), allocatable :: eventsrc[:]
  type(lock_type), allocatable :: locksrc(:)[:]
  type(recursiveType), allocatable :: recoksrc
  class(recursiveTypeNok), allocatable :: recnoksrc[:]

  ! Valid constructs
  allocate(okt1[*], SOURCE=okt1src)
  allocate(okt2[*], SOURCE=okt2src)
  allocate(okt3[*], SOURCE=okt3src)
  allocate(whatever[*], SOURCE=okt3src)
  allocate(recok, SOURCE=recoksrc)

  allocate(nokt1[*])
  allocate(nokt2[*])
  allocate(nokt3[*])
  allocate(nokt4[*])
  allocate(nokt5[*])
  allocate(nokt6(10)[*])
  allocate(lock(10)[*])
  allocate(event[*])
  allocate(recnok[*])

  allocate(nokt1[*], MOLD=nokt1src)
  allocate(nokt2[*], MOLD=nokt2src)
  allocate(nokt3[*], MOLD=nokt3src)
  allocate(nokt4[*], MOLD=nokt4src)
  allocate(nokt5[*], MOLD=nokt5src)
  allocate(nokt6[*], MOLD=nokt6src)
  allocate(lock[*],  MOLD=locksrc)
  allocate(event[*], MOLD=eventsrc)
  allocate(recnok[*],MOLD=recnoksrc)
  allocate(whatever[*],MOLD=nokt6src)

  !ERROR: SOURCE expression type must not have potential subobject component of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(nokt1[*], SOURCE=nokt1src)
  !ERROR: SOURCE expression type must not have potential subobject component of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(nokt2[*], SOURCE=nokt2src)
  !ERROR: SOURCE expression type must not have potential subobject component of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(nokt3[*], SOURCE=nokt3src)
  !ERROR: SOURCE expression type must not have potential subobject component of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(nokt4[*], SOURCE=nokt4src)
  !ERROR: SOURCE expression type must not have potential subobject component of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(nokt5[*], SOURCE=nokt5src)
  !ERROR: SOURCE expression type must not have potential subobject component of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(nokt6[*], SOURCE=nokt6src)
  !ERROR: SOURCE expression type must not be EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(lock[*],  SOURCE=locksrc)
  !ERROR: SOURCE expression type must not be EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(event[*], SOURCE=eventsrc)
  !ERROR: SOURCE expression type must not have potential subobject component of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(recnok[*],SOURCE=recnoksrc)
  !ERROR: SOURCE expression type must not have potential subobject component of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV
  allocate(whatever[*],SOURCE=nokt5src)
end subroutine


subroutine C948_b()
  use not_iso_fortran_env !type restriction do not apply

  type oktype1
    type(event_type), allocatable :: event
  end type

  type oktype2
    type(lock_type) :: lock
  end type

  type(oktype1), allocatable :: okt1[:]
  class(oktype2), allocatable :: okt2[:]
  type(event_type), allocatable :: team[:]
  class(lock_type), allocatable :: lock[:]

  type(oktype1), allocatable :: okt1src[:]
  class(oktype2), allocatable :: okt2src[:]
  class(event_type), allocatable :: teamsrc[:]
  type(lock_type), allocatable :: locksrc[:]

  allocate(okt1[*], SOURCE=okt1src)
  allocate(okt2[*], SOURCE=okt2src)
  allocate(team[*], SOURCE=teamsrc)
  allocate(lock[*], SOURCE=locksrc)
end subroutine
