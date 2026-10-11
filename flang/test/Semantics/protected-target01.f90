! RUN: %python %S/test_errors.py %s %flang_fc1
! Declaration and allocation constraints from J3/26-007r1.

subroutine declarations
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  integer, protected_target :: not_a_pointer
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  integer, allocatable, protected_target :: allocatable_object
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  integer :: statement_object
  protected_target statement_object
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  integer :: statement_object2
  protected_target :: statement_object2
  !ERROR: A PROTECTED_TARGET pointer may not be in a common block
  integer, pointer, protected_target :: common_pointer
  common /block/ common_pointer
  integer, pointer :: p, q
  protected_target p, q
  integer :: forward_pointer
  protected_target :: forward_pointer
  pointer :: forward_pointer
  type bad
    !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
    integer, protected_target :: component
    integer, pointer, protected_target :: good => null()
  end type
  nullify(p, q, forward_pointer)
end subroutine

subroutine common_components
  type inner
    sequence
    integer, pointer, protected_target :: p
  end type
  type outer
    sequence
    type(inner) :: nested
  end type
  type(outer) :: object
  !ERROR: Object 'object' in COMMON block /components/ has PROTECTED_TARGET pointer component 'p'
  common /components/ object
  type(outer), pointer :: ordinary
  common /pointers/ ordinary
end subroutine

subroutine statement_constraint
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  real :: x
  protected_target x
end subroutine

subroutine procedure_constraint
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  procedure(), pointer :: proc_pointer
  protected_target :: proc_pointer
end subroutine

subroutine allocation
  type t
    integer, pointer, protected_target :: component
  end type
  type(t) :: object
  integer, pointer, protected_target :: p, a(:)
  integer, pointer :: ordinary
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(p)
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(p, mold=1)
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(integer :: p)
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(object%component)
  allocate(p, source=1)
  allocate(a(2), source=1)
  allocate(object%component, source=1)
  !ERROR: Name in DEALLOCATE statement is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer and may not be deallocated
  deallocate(p)
  !ERROR: Name in DEALLOCATE statement is not definable
  !BECAUSE: 'component' is a PROTECTED_TARGET pointer and may not be deallocated
  deallocate(object%component)
  nullify(p, a, object%component)
  allocate(ordinary)
  ordinary = 1
  deallocate(ordinary)
end subroutine

subroutine intent_in(p, ordinary)
  integer, pointer, protected_target, intent(in) :: p
  integer, pointer, intent(in) :: ordinary
  integer, target :: x
  !ERROR: 'p' may not appear in NULLIFY
  !BECAUSE: 'p' is an INTENT(IN) dummy argument
  nullify(p)
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'p' is an INTENT(IN) dummy argument
  p => x
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
  p = 2
  ordinary = 2
end subroutine
