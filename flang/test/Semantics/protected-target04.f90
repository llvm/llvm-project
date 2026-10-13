! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable
! Follow F2028 draft C864 rather than the conflicting 8.5.16 NOTE 1 example.
! Ordinary pointers retain their optional diagnostic.

module protected_target_conflict
  integer, target, protected :: value = 1
  integer, pointer, protected :: ordinary => value
  integer, pointer, protected, protected_target :: protected => value
contains
  subroutine owner
    integer, pointer, protected_target :: p
    p => value
    protected => value
    ordinary = 2
    print *, p
  end subroutine
end module

subroutine use_protected_target
  use protected_target_conflict
  integer, pointer :: q
  integer, pointer, protected_target :: p
  !ERROR: A PROTECTED nonpointer target 'value' is USE-associated and may not be associated with a PROTECTED_TARGET pointer
  p => value
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'value' is protected in this scope
  q => value
  p => ordinary
  ordinary = 2
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'protected' is protected in this scope
  protected => null()
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'protected' is a PROTECTED_TARGET pointer whose target is not definable
  protected = 2
end subroutine
