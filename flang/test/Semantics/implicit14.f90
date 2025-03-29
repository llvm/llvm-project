! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type dt
    procedure(explicit), pointer, nopass :: p
  end type
 contains
  integer function one()
    one = 1
  end
  function onePtr()
    procedure(one), pointer :: onePtr
    onePtr => one
  end
  function explicit
    character(:), allocatable :: explicit
    explicit = "abc"
  end
end

program test
  use m
  procedure(), pointer :: p0
  procedure(one), pointer :: p1
  procedure(integer), pointer :: p2
  procedure(explicit), pointer :: p3
  external implicit
  type(dt) x
  p0 => one ! ok
  p0 => onePtr() ! ok
  p0 => implicit ! ok
  !ERROR: Procedure pointer 'p0' with implicit interface may not be associated with procedure designator 'explicit' with explicit interface that cannot be called via an implicit interface
  p0 => explicit
  p1 => one ! ok
  p1 => onePtr() ! ok
  p1 => implicit ! ok
  !ERROR: Function pointer 'p1' associated with incompatible function designator 'explicit': function results have incompatible attributes
  p1 => explicit
  p2 => one ! ok
  p2 => onePtr() ! ok
  p2 => implicit ! ok
  !ERROR: Function pointer 'p2' associated with incompatible function designator 'explicit': function results have incompatible attributes
  p2 => explicit
  !ERROR: Function pointer 'p3' associated with incompatible function designator 'one': function results have incompatible attributes
  p3 => one
  !ERROR: Procedure pointer 'p3' associated with result of reference to function 'oneptr' that is an incompatible procedure pointer: function results have incompatible attributes
  p3 => onePtr()
  p3 => explicit ! ok
  !ERROR: Procedure pointer 'p3' with explicit interface that cannot be called via an implicit interface cannot be associated with procedure designator with an implicit interface
  p3 => implicit
  !ERROR: Procedure pointer 'p' with explicit interface that cannot be called via an implicit interface cannot be associated with procedure designator with an implicit interface
  x = dt(implicit)
  !ERROR: Procedure pointer 'p' with explicit interface that cannot be called via an implicit interface cannot be associated with procedure designator with an implicit interface
  x%p => implicit
end
