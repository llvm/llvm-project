! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 C866, C868, C869: allocation changes association, not existing targets.

module protected_target_allocation
  type t
    integer, allocatable :: a
    integer, pointer :: p
    integer, pointer, protected_target :: r
  end type
contains
  subroutine allocation(p, q, obj, ordinary)
    integer, pointer, protected_target :: p, q(:)
    type(t), pointer, protected_target :: obj
    type(t) :: ordinary
    integer, allocatable :: a
    !ERROR: PROTECTED_TARGET pointer in ALLOCATE must have a SOURCE= specifier
    allocate(p)
    !ERROR: PROTECTED_TARGET pointer in ALLOCATE must have a SOURCE= specifier
    allocate(p, mold=1)
    allocate(p, source=1)
    allocate(q(2), source=[1, 2])
    !ERROR: PROTECTED_TARGET pointer may not appear in DEALLOCATE
    deallocate(p)
    !ERROR: PROTECTED_TARGET pointer in ALLOCATE must have a SOURCE= specifier
    allocate(ordinary%r)
    allocate(ordinary%r, source=1)
    !ERROR: PROTECTED_TARGET pointer may not appear in DEALLOCATE
    deallocate(ordinary%r)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    allocate(obj%a)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    allocate(obj%p, source=1)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    allocate(obj%r, source=1)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    deallocate(obj%a)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    deallocate(obj%p)
    !ERROR: 'p' may not appear in NULLIFY
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    nullify(obj%p)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'from=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    call move_alloc(obj%a, a)
    allocate(ordinary%a)
    deallocate(ordinary%a)
    nullify(p, q)
  end subroutine

  subroutine intent_in(p)
    integer, pointer, protected_target, intent(in) :: p
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    allocate(p, source=1)
  end subroutine
end module
