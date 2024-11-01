! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for the ALLOCATED() intrinsic
subroutine alloc(coarray_alloc, coarray_not_alloc, t2_not_alloc, &
                 assumedRank)

  interface
    function return_allocatable()
      integer, allocatable :: return_allocatable(:)
    end function
  end interface

  type :: t1
    integer, allocatable :: alloc(:)
    integer :: not_alloc
  end type

  type :: t2
    real, allocatable :: coarray_alloc[:]
    real, allocatable :: coarray_alloc_array(:)[:]
  end type


  integer :: not_alloc(100)
  real, allocatable :: x_alloc
  character(:), allocatable :: char_alloc(:)
  type(t1) :: dt_not_alloc(100)
  type(t1), allocatable :: dt_alloc(:)

  real, allocatable :: coarray_alloc[:, :]
  real, allocatable :: coarray_alloc_array(:)[:, :]
  real :: coarray_not_alloc(:)[*]

  type(t2) :: t2_not_alloc
  real, allocatable :: assumedRank(..)


  ! OK
  print *, allocated(x_alloc)
  print *, allocated(char_alloc)
  print *, allocated(dt_alloc)
  print *, allocated(dt_not_alloc(3)%alloc)
  print *, allocated(dt_alloc(3)%alloc)
  print *, allocated(coarray_alloc)
  print *, allocated(coarray_alloc[2,3])
  print *, allocated(t2_not_alloc%coarray_alloc)
  print *, allocated(t2_not_alloc%coarray_alloc[2])
  print *, allocated(assumedRank)
  select rank (assumedRank)
  rank (0)
    print *, allocated(scalar=assumedRank)
  rank default
    print *, allocated(array=assumedRank)
  end select

  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(not_alloc)
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(dt_not_alloc)
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(dt_alloc%not_alloc)
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(char_alloc(:))
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(char_alloc(1)(1:10))
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(coarray_alloc_array(1:10))
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(coarray_alloc_array(1:10)[2,2])
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(t2_not_alloc%coarray_alloc_array(1))
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(t2_not_alloc%coarray_alloc_array(1)[2])
  !ERROR: Argument of ALLOCATED() must be an ALLOCATABLE object or component
  print *, allocated(return_allocatable())
end subroutine
