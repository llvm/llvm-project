! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable
! F'2028 8.5.16 NOTE 1 conflicts with C864. Follow NOTE 1 for
! PROTECTED_TARGET pointers, without changing ordinary pointer diagnostics.

module protected_targets
  integer, protected, target :: x
  integer, pointer, protected_target :: p
  integer, protected, pointer :: ordinary
contains
  subroutine owner
    p => x
    ordinary => x
    ordinary = 1 ! PROTECTED protects association, not target definition.
  end subroutine
end module

subroutine use_targets
  use protected_targets
  integer, pointer :: q, w
  p => x ! Follow NOTE 1, including when optional warnings are enabled.
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  q => x
  associate (a => p)
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    w => a
  end associate
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  p = 999
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'ordinary' is protected in this scope
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  ordinary => x
  ordinary = 1
end subroutine

subroutine intent_target(x, ordinary)
  integer, target, intent(in) :: x
  integer, pointer, intent(in) :: ordinary
  integer, pointer, protected_target :: p
  integer, pointer :: q
  p => x
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is an INTENT(IN) dummy argument
  q => x
  ordinary = 1 ! INTENT(IN) protects association, not target definition.
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'ordinary' is an INTENT(IN) dummy argument
  ordinary => null()
end subroutine

subroutine constant_target
  integer, parameter :: x = 1
  integer, pointer, protected_target :: p
  !ERROR: Target associated with pointer 'p' must be a designator or a call to a pointer-valued function
  p => x
end subroutine
