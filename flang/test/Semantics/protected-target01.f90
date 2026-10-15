! RUN: %python %S/test_errors.py %s %flang_fc1
! F'2028 C781 and C8109: only data pointers may have PROTECTED_TARGET.
! C867 prohibits PROTECTED_TARGET pointers in COMMON.

subroutine declarations
  type t
    !ERROR: PROTECTED_TARGET entity 'c' must be a data pointer
    integer, protected_target :: c
    integer, pointer, protected_target :: p
  end type
  !ERROR: PROTECTED_TARGET entity 'scalar' must be a data pointer
  integer, protected_target :: scalar
  !ERROR: PROTECTED_TARGET entity 'alloc' must be a data pointer
  integer, allocatable, protected_target :: alloc
  !ERROR: PROTECTED_TARGET entity 'statement_scalar' must be a data pointer
  integer :: statement_scalar
  protected_target statement_scalar
  !ERROR: PROTECTED_TARGET entity 'proc' must be a data pointer
  procedure(), pointer :: proc
  protected_target :: proc
  integer, pointer :: p, q, r
  protected_target p, q
  protected_target :: r
  protected_target :: before_declaration
  integer, pointer :: before_declaration
  block
    integer, pointer :: local
    protected_target local
    nullify(local)
  end block
end subroutine

subroutine common_pointer_target
  type t
    sequence
    integer, pointer, protected_target :: p
  end type
  type(t), pointer :: x
  common /ok/ x ! Only x's descriptor is in COMMON, not its target's components.
end subroutine

subroutine common_pointer
  !ERROR: A PROTECTED_TARGET pointer may not be in a common block
  integer, pointer, protected_target :: p
  common /bad/ p
end subroutine

subroutine common_component
  type t
    sequence
    integer, pointer, protected_target :: p
  end type
  type(t) :: x
  common /ok_component/ x ! C867 restricts the pointer itself, not its container.
end subroutine

module protected_target_function_names
contains
  !ERROR: PROTECTED_TARGET entity 'bad_function' must be a data pointer
  function bad_function() result(res)
    integer, pointer :: res
    protected_target :: bad_function
    nullify(res)
  end function
  function good_function()
    integer, pointer :: good_function
    protected_target :: good_function
    nullify(good_function)
  end function
  function good_result() result(res)
    integer, pointer :: res
    protected_target :: res
    nullify(res)
  end function
end module
