! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 C781, C867-C869 and C8109.

subroutine declarations
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  integer, protected_target :: nonpointer
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  integer, allocatable, protected_target :: allocatable_object
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  integer :: statement_object
  protected_target statement_object
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  procedure(real), pointer :: procedure_pointer
  protected_target :: procedure_pointer
  !ERROR: A PROTECTED_TARGET pointer may not be in a common block
  integer, pointer, protected_target :: common_pointer
  common /block/ common_pointer
  type t
    !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
    integer, protected_target :: component
  end type
end subroutine

subroutine data_pointer_not_external
  !ERROR: 'f' is a data object and may not be EXTERNAL
  real, pointer, protected_target :: f
  !ERROR: EXTERNAL attribute not allowed on 'f'
  external f
end subroutine

subroutine common_component
  type t
    sequence
    integer, pointer, protected_target :: component
  end type
  type(t) :: object
  type(t), pointer :: ordinary_pointer
  !ERROR: COMMON block /bad/ may not contain a PROTECTED_TARGET pointer component 'component'
  common /bad/ object
  common /good/ ordinary_pointer
end subroutine

subroutine statement_constraint
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  integer :: nonpointer
  protected_target :: nonpointer
end subroutine

!ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
function protected_function_name() result(r)
  integer, pointer :: r
  protected_target :: protected_function_name
end function

subroutine allocation
  type t
    integer, pointer, protected_target :: component
  end type
  type(t) :: object
  integer, pointer, protected_target :: p, a(:)
  integer, pointer :: ordinary
  allocate(p, source=42)
  allocate(a(2), source=[1, 2])
  allocate(object%component, source=42)
  allocate(ordinary)
  deallocate(ordinary)
  !ERROR: Allocation of a PROTECTED_TARGET pointer requires SOURCE=
  allocate(p)
  !ERROR: Allocation of a PROTECTED_TARGET pointer requires SOURCE=
  allocate(p, mold=42)
  !ERROR: Allocation of a PROTECTED_TARGET pointer requires SOURCE=
  allocate(integer :: p)
  !ERROR: Allocation of a PROTECTED_TARGET pointer requires SOURCE=
  allocate(object%component)
  !ERROR: A PROTECTED_TARGET pointer may not be deallocated
  deallocate(p)
  !ERROR: A PROTECTED_TARGET pointer may not be deallocated
  deallocate(object%component)
  nullify(p, a, object%component)
end subroutine
