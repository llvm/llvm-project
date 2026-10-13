! RUN: %python %S/test_errors.py %s %flang_fc1
! Test that DEALLOCATE works

INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt),Pointer :: p
End Type

Type(t),Allocatable :: x(:)
Type(t),Pointer :: y(:)
Type(t),Pointer :: z
Integer :: s
CHARACTER(256) :: e

Integer, Pointer :: pi

Allocate(pi)
Allocate(x(3))

Deallocate(x(2)%p)

Deallocate(y(2)%p)

Deallocate(pi)

Deallocate(z%p)

!ERROR: An allocatable or pointer component reference must be applied to a scalar base
Deallocate(x%p, stat=s, errmsg=e)
Deallocate(x, errmsg=e)
Deallocate(x, stat=s)

Deallocate(y, stat=s, errmsg=e)
Deallocate(y, errmsg=e)
Deallocate(y, stat=s)

Deallocate(z, stat=s, errmsg=e)
Deallocate(z, errmsg=e)
Deallocate(z, stat=s)

Deallocate(z, y, stat=s, errmsg=e)
Deallocate(z, y, errmsg=e)
Deallocate(z, y, stat=s)

End Program

! J3/26-007r1 C868 and target/subobject definition.
subroutine protected_target_deallocation
  type node
    real, pointer :: link
    type(node), pointer :: next
    real, pointer, protected_target :: protected_link
  end type
  real, pointer, protected_target :: p
  type(node) :: object
  type(node), pointer, protected_target :: parent
  !ERROR: A PROTECTED_TARGET pointer may not appear in a DEALLOCATE statement
  deallocate(p)
  !ERROR: A PROTECTED_TARGET pointer may not appear in a DEALLOCATE statement
  deallocate(object%protected_link)
  !ERROR: Name in DEALLOCATE statement is not definable
  !BECAUSE: 'parent' has the PROTECTED_TARGET attribute and its target is not definable
  deallocate(parent%link)
  deallocate(parent%next%link) ! The descriptor belongs to a different target.
  associate(q => parent)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: 'parent' has the PROTECTED_TARGET attribute and its target is not definable
    deallocate(q%link)
  end associate
end subroutine

! J3/26-007r1 C866 and 20.6.7(9),(11).
subroutine protected_target_rank_deallocation(p)
  real, pointer, protected_target :: p(..)
  select rank(q => p)
  rank(0)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
    deallocate(q)
  rank(1)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
    deallocate(q)
  end select
end subroutine

subroutine ordinary_rank_deallocation(p)
  real, pointer :: p(..)
  select rank(q => p)
  rank(0)
    deallocate(q)
  rank(1)
    deallocate(q)
  end select
end subroutine
