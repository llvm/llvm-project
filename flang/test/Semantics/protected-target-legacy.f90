! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable
! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT --implicit-check-not=warning:
! Ordinary-pointer INTENT(IN) and PROTECTED checks, including target warnings.
! DEFAULT: error: 'in_pointer' may not appear in NULLIFY

module legacy
  integer, target, protected :: protected_value
  integer, pointer, protected :: protected_pointer
contains
  subroutine owner
    protected_pointer => protected_value
    protected_value = 1
  end subroutine
end module

subroutine test(in_pointer, in_target)
  use legacy
  integer, pointer, intent(in) :: in_pointer
  integer, target, intent(in) :: in_target
  integer, pointer :: ordinary
  integer, pointer, protected_target :: readonly
  in_pointer = 1
  protected_pointer = 1
  !ERROR: 'in_pointer' may not appear in NULLIFY
  !BECAUSE: 'in_pointer' is an INTENT(IN) dummy argument
  nullify(in_pointer)
  !ERROR: 'protected_pointer' may not appear in NULLIFY
  !BECAUSE: 'protected_pointer' is protected in this scope
  nullify(protected_pointer)
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'in_target' is an INTENT(IN) dummy argument
  ordinary => in_target
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'protected_value' is protected in this scope
  ordinary => protected_value
  readonly => in_target
  ! Follow F2028 8.5.16 NOTE 1 despite the draft's conflicting C864.
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'protected_value' is protected in this scope
  readonly => protected_value
end subroutine
