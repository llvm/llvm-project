! RUN: %python %S/test_errors.py %s %flang_fc1
! Every other Fortran compiler (but one) interprets the names of parent
! components like this when the names of their types are the product of
! USE association with renaming.

module m1
  type originalName
    integer m
  end type
end

module m2
  use m1, newName => originalName
  type, extends(newName) :: extended
    integer n
  end type
  type, extends(newName) :: extended2
    integer originalName ! ok
  end type
 contains
  subroutine s1
    type(extended) x
    type(extended2) x2
    print *, x%newName%m ! ok
    !ERROR: Component 'originalname' not found in derived type 'extended'
    print *, x%originalName
    print *, extended(newName=newName(m=1),n=2) ! ok
    !ERROR: Structure constructor lacks a value for component 'm'
    !ERROR: Keyword 'originalname=' does not name a component of derived type 'extended'
    !ERROR: Keyword 'm=' may not appear in a reference to a procedure with an implicit interface
    print *, extended(originalName=originalName(m=1),n=2)
    !ERROR: Value in structure constructor of type 'REAL(4)' is incompatible with component 'newname' of type 'newname'
    !ERROR: Keyword 'm=' may not appear in a reference to a procedure with an implicit interface
    print *, extended(newName=originalName(m=1),n=2)
    !ERROR: Structure constructor lacks a value for component 'm'
    !ERROR: Keyword 'originalname=' does not name a component of derived type 'extended'
    print *, extended(originalName=newName(m=1),n=2)
    print *, x2%newName%m ! ok
    print *, x2%originalName ! ok
    print *, extended2(newName=newName(m=1),originalName=2) ! ok
  end
end

module m3
  use m2
 contains
  ! Same as above, but not in the same module as the derived
  ! types' definitions.
  subroutine s2
    type(extended) x
    type(extended2) x2
    print *, x%newName%m ! ok
    !ERROR: Component 'originalname' not found in derived type 'extended'
    print *, x%originalName
    print *, extended(newName=newName(m=1),n=2) ! ok
    !ERROR: Structure constructor lacks a value for component 'm'
    !ERROR: Keyword 'originalname=' does not name a component of derived type 'extended'
    !ERROR: Keyword 'm=' may not appear in a reference to a procedure with an implicit interface
    print *, extended(originalName=originalName(m=1),n=2)
    !ERROR: Value in structure constructor of type 'REAL(4)' is incompatible with component 'newname' of type 'newname'
    !ERROR: Keyword 'm=' may not appear in a reference to a procedure with an implicit interface
    print *, extended(newName=originalName(m=1),n=2)
    !ERROR: Structure constructor lacks a value for component 'm'
    !ERROR: Keyword 'originalname=' does not name a component of derived type 'extended'
    print *, extended(originalName=newName(m=1),n=2)
    print *, x2%newName%m ! ok
    print *, x2%originalName ! ok
    print *, extended2(newName=newName(m=1),originalName=2) ! ok
  end
end
