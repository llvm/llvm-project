! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable -Werror

! This captures the current working-draft conflict between 8.5.15 C864 and
! 8.5.16 NOTE 1. Flang accepts association through a PROTECTED_TARGET pointer
! by default; the optional warning remains for ordinary pointers.

module m_conflict
  implicit none
  integer, target, protected :: x = 1
end module

module m_conflict_use
  use m_conflict
  implicit none
  type t
    integer, pointer, protected_target :: c => x
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'x' is protected in this scope
    integer, pointer :: d => x
  end type
  integer, pointer, protected_target :: init_pt => x
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  integer, pointer :: init_plain => x
contains
  subroutine outside
    type(t) :: o
    integer, pointer, protected_target :: p
    integer, pointer :: q
    p => x
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'x' is protected in this scope
    q => x
    o%c => x
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'x' is protected in this scope
    o%d => x
    o = t(x, null())
  end
end module
