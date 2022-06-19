! RUN: %python %S/test_errors.py %s %flang_fc1
! Error tests for recursive use of derived types.
! C744 If neither the POINTER nor the ALLOCATABLE attribute is specified, the
! declaration-type-spec in the component-def-stmt shall specify an intrinsic
! type or a previously defined derived type.

program main
  type :: recursive1
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    type(recursive1) :: bad1
    type(recursive1), pointer :: ok1
    type(recursive1), allocatable :: ok2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    !ERROR: CLASS entity 'bad2' must be a dummy argument or have ALLOCATABLE or POINTER attribute
    class(recursive1) :: bad2
    class(recursive1), pointer :: ok3
    class(recursive1), allocatable :: ok4
  end type recursive1
  type :: recursive2(kind,len)
    integer, kind :: kind
    integer, len :: len
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    type(recursive2(kind,len)) :: bad1
    type(recursive2(kind,len)), pointer :: ok1
    type(recursive2(kind,len)), allocatable :: ok2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    !ERROR: CLASS entity 'bad2' must be a dummy argument or have ALLOCATABLE or POINTER attribute
    class(recursive2(kind,len)) :: bad2
    class(recursive2(kind,len)), pointer :: ok3
    class(recursive2(kind,len)), allocatable :: ok4
  end type recursive2
  type :: recursive3(kind,len)
    integer, kind :: kind = 1
    integer, len :: len = 2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    type(recursive3) :: bad1
    type(recursive3), pointer :: ok1
    type(recursive3), allocatable :: ok2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    !ERROR: CLASS entity 'bad2' must be a dummy argument or have ALLOCATABLE or POINTER attribute
    class(recursive3) :: bad2
    class(recursive3), pointer :: ok3
    class(recursive3), allocatable :: ok4
  end type recursive3
  !ERROR: Derived type 'recursive4' cannot extend itself
  type, extends(recursive4) :: recursive4
  end type recursive4
end program main
