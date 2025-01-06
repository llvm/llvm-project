! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! A construct entity does not have the POINTER or ALLOCATABLE attribute,
! except in SELECT RANK.

subroutine test(up,ua,rp,ra)
  class(*), pointer :: up
  class(*), allocatable :: ua
  real, pointer :: rp(..)
  real, allocatable :: ra(..)
  real, target :: x
  real, pointer :: p
  real, allocatable :: a
  associate (s => p)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 's' is not a pointer
    s => x
    !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    allocate(s)
    !ERROR: Name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    deallocate(s)
    !ERROR: 's' may not appear in NULLIFY
    !BECAUSE: 's' is not a pointer
    nullify(s)
  end associate
  select type(s => up)
  type is (real)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 's' is not a pointer
    s => x
    !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    allocate(s)
    !ERROR: Name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    deallocate(s)
    !ERROR: 's' may not appear in NULLIFY
    !BECAUSE: 's' is not a pointer
    nullify(s)
  end select
  select rank(s => rp)
  rank(0)
    s => x ! ok
    allocate(s) ! ok
    deallocate(s) ! ok
    nullify(s) ! ok
  !ERROR: RANK (*) cannot be used when selector is POINTER or ALLOCATABLE
  rank(*)
  rank default
    !ERROR: The left-hand side of a pointer assignment must not be an assumed-rank dummy argument
    !ERROR: pointer 's' associated with object 'x' with incompatible type or shape
    s => x
    !ERROR: An assumed-rank dummy argument may not appear in an ALLOCATE statement
    allocate(s)
    deallocate(s) ! ok
    nullify(s) ! ok
  end select
  associate (s => a)
    !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    allocate(s)
    !ERROR: Name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    deallocate(s)
  end associate
  select type(s => ua)
  type is (real)
    !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    allocate(s)
    !ERROR: Name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    deallocate(s)
  end select
  select rank(s => ra)
  rank(0)
    allocate(s) ! ok
    deallocate(s) ! ok
  !ERROR: RANK (*) cannot be used when selector is POINTER or ALLOCATABLE
  rank(*)
  rank default
    !ERROR: An assumed-rank dummy argument may not appear in an ALLOCATE statement
    allocate(s)
    deallocate(s) ! ok
  end select
end
