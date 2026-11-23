! RUN: %python %S/test_errors.py %s %flang_fc1
! A CLASS() entity must be a dummy argument, allocatable,
! or object pointer.  Don't get confused with procedure pointers.
module m
  type t
  end type
  !ERROR: CLASS entity 'v1' must be a dummy argument, allocatable, or object pointer
  class(t) v1
  class(t), allocatable :: v2 ! ok
  class(t), pointer :: v3 ! ok
  !ERROR: Polymorphic function 'p1' must have an explicit interface whose result is ALLOCATABLE or POINTER
  procedure(cf1) :: p1
  procedure(cf2) :: p2
  procedure(cf3) :: p3
  !ERROR: Polymorphic function pointer 'pp1' must have an explicit interface whose result is ALLOCATABLE or POINTER
  procedure(cf1), pointer :: pp1
  procedure(cf2), pointer :: pp2
  procedure(cf3), pointer :: pp3
  procedure(cf5), pointer :: pp4 ! ok
 contains
  !ERROR: CLASS entity 'cf1' must be a dummy argument, allocatable, or object pointer
  class(t) function cf1()
  end
  class(t) function cf2()
    allocatable cf2 ! ok
  end
  class(t) function cf3()
    pointer cf3 ! ok
  end
  subroutine test(d1,d2,d3)
    class(t) d1 ! ok
    !ERROR: Polymorphic function 'd2' must have an explicit interface whose result is ALLOCATABLE or POINTER
    class(t), external :: d2
    !ERROR: Polymorphic function pointer 'd3' must have an explicit interface whose result is ALLOCATABLE or POINTER
    class(t), external, pointer :: d3
  end
  function cf4()
    class(t), pointer :: cf4
    cf4 => v3
  end
  function cf5
    procedure(cf4), pointer :: cf5
    cf5 => cf4
  end
end
