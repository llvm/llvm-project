! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable
! J3/26-007r1 C864 conflicts with 8.5.16 NOTE 1. Follow NOTE 1 for
! PROTECTED_TARGET; retain the ordinary-pointer warning and target semantics.

module protected_owner
  integer, protected, target :: x
  integer, protected, pointer :: ordinary_protected
contains
  subroutine owner
    integer, pointer, protected_target :: p
    p => x
    nullify(p)
  end subroutine
end module

subroutine use_protected
  use protected_owner
  integer, pointer :: ordinary
  integer, target :: local
  integer, pointer, protected_target :: p, q
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  ordinary => x
  p => x
  q => p
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
  p = 1
  ordinary_protected = 1
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'ordinary_protected' is protected in this scope
  ordinary_protected => local
end subroutine

subroutine intent_target(x)
  integer, target, intent(in) :: x
  integer, pointer, protected_target :: p
  integer, pointer :: ordinary
  p => x
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is an INTENT(IN) dummy argument
  ordinary => x
  ordinary = 1
  nullify(p, ordinary)
end subroutine
