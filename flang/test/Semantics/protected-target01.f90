! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 C781, C867, C8109: declarations and COMMON.

module protected_target_declarations
  type t
    integer, pointer, protected_target :: p => null()
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    integer, protected_target :: not_pointer
  end type
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  integer, protected_target :: not_pointer
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  integer :: statement_not_pointer
  protected_target statement_not_pointer
  ! The attribute statement does not make a procedure pointer a data pointer.
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  procedure(), pointer :: procedure_pointer
  protected_target :: procedure_pointer
  integer, pointer :: p, q
  protected_target :: p
  protected_target q
  integer, target :: x
contains
  function get() result(r)
    integer, pointer, protected_target :: r
    r => x
  end function
  subroutine in_common
    !ERROR: A PROTECTED_TARGET entity may not be in a common block
    integer, pointer, protected_target :: c
    common /block/ c
  end subroutine
end module
