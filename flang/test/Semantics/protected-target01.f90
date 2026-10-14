! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 draft J3/26-007r1 C781, C867, C8109 and data-pointer eligibility.

module declarations
  implicit none
  type good
    integer, pointer, protected_target :: p
    integer, protected_target, pointer :: q
  end type
  type bad
    !ERROR: PROTECTED_TARGET entity 'x' must be a data pointer
    integer, protected_target :: x
    !ERROR: PROTECTED_TARGET entity 'a' must be a data pointer
    integer, allocatable, protected_target :: a
  end type
  !ERROR: PROTECTED_TARGET entity 'nonpointer' must be a data pointer
  integer, protected_target :: nonpointer
  !ERROR: PROTECTED_TARGET entity 'constant' must be a data pointer
  integer, parameter, protected_target :: constant = 1
  !ERROR: PROTECTED_TARGET entity 'proc' must be a data pointer
  procedure(sin), pointer :: proc
  protected_target proc
  protected_target :: statement_object
  !ERROR: PROTECTED_TARGET entity 'statement_object' must be a data pointer
  integer :: statement_object
contains
  subroutine statement_order
    protected_target p
    integer p
    pointer p
    integer, pointer :: q
    protected_target :: q
    integer, target :: x
    p => x
    q => p
    nullify(p, q)
  end subroutine
  function result_pointer() result(p)
    integer, pointer :: p
    protected_target p
    nullify(p)
  end function
end module

subroutine common_pointer
  integer, pointer, protected_target :: p
  !ERROR: PROTECTED_TARGET pointer 'p' may not appear in COMMON block /pt_block/
  common /pt_block/ p
end

subroutine common_component
  type inner
    sequence
    integer, pointer, protected_target :: p
  end type
  type outer
    sequence
    type(inner) :: nested
  end type
  type(outer) :: x
  ! C867 applies to the named COMMON object, not its pointer components.
  common /pt_component/ x
end

subroutine common_ordinary
  type inner
    sequence
    integer, pointer, protected_target :: p
  end type
  integer, pointer :: ordinary
  type(inner), pointer :: separate_target
  common /ordinary_block/ ordinary, separate_target
end
