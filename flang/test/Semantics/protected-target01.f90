! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 draft C781, C8109, C866-C869 and 8.5.16p2.

subroutine declarations
  type bad
    !ERROR: PROTECTED_TARGET component 'component' must have the POINTER attribute
    integer, protected_target :: component
  end type
  !ERROR: PROTECTED_TARGET entity 'nonpointer' must have the POINTER attribute
  integer, protected_target :: nonpointer
  !ERROR: PROTECTED_TARGET entity 'statement_nonpointer' must have the POINTER attribute
  integer :: statement_nonpointer
  protected_target :: statement_nonpointer
  !ERROR: PROTECTED_TARGET entity 'proc_pointer' must be a data pointer
  procedure(), pointer :: proc_pointer
  protected_target proc_pointer
  integer :: forward_pointer
  protected_target forward_pointer
  pointer :: forward_pointer
end subroutine

subroutine common_block
  !ERROR: A PROTECTED_TARGET pointer may not be in a common block
  integer, pointer, protected_target :: p
  common /block/ p
end subroutine

subroutine common_component
  type inner
    sequence
    integer, pointer, protected_target :: p
  end type
  type outer
    sequence
    type(inner) :: component
  end type
  type(outer) :: object
  !ERROR: COMMON block /block/ may not have the member 'object' with PROTECTED_TARGET component 'p'
  common /block/ object
end subroutine

subroutine common_pointer_target
  type box
    sequence
    integer, pointer, protected_target :: p
  end type
  ! A common pointer's target is not stored in the common block.
  type(box), pointer :: ordinary
  common /block/ ordinary
end subroutine

module protected_target_definitions
  type box
    integer :: value
    integer, allocatable :: alloc(:)
    integer, pointer :: ordinary(:)
    integer, pointer, protected_target :: protected(:)
  end type
  type initialized
    integer :: value = 1
  end type
contains
  subroutine definitions(p, a, b, c, z)
    integer, pointer, protected_target :: p, a(:)
    type(box), pointer, protected_target :: b
    character(*), pointer, protected_target :: c
    complex, pointer, protected_target :: z
    integer :: value
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
    p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
    if (.false.) p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'a' is a PROTECTED_TARGET pointer whose target is not definable
    a(1) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'a' is a PROTECTED_TARGET pointer whose target is not definable
    a(1:2) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'a' is a PROTECTED_TARGET pointer whose target is not definable
    a([1,2]) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'b' is a PROTECTED_TARGET pointer whose target is not definable
    b%value = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'c' is a PROTECTED_TARGET pointer whose target is not definable
    c(1:2) = 'ab'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'z' is a PROTECTED_TARGET pointer whose target is not definable
    z%re = 1
    !ERROR: Input variable 'p' is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
    read(*,*) p
    !ERROR: IOSTAT variable 'p' is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
    read(*,*,iostat=p) value
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'protected' is a PROTECTED_TARGET pointer whose target is not definable
    b%protected = 1
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'b' is a PROTECTED_TARGET pointer whose target is not definable
    b%ordinary => null()
    !ERROR: 'ordinary' may not appear in NULLIFY
    !BECAUSE: 'b' is a PROTECTED_TARGET pointer whose target is not definable
    nullify(b%ordinary)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: 'b' is a PROTECTED_TARGET pointer whose target is not definable
    allocate(b%alloc(2), source=1)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: 'b' is a PROTECTED_TARGET pointer whose target is not definable
    deallocate(b%alloc)
    !ERROR: A PROTECTED_TARGET pointer may not appear in DEALLOCATE
    deallocate(b%protected)
    ! A nested ordinary pointer's target is an independent object.
    b%ordinary = 1
    print *, p, a, b%value, c, z
    nullify(p, a, b, c, z)
  end subroutine

  subroutine allocation
    integer, pointer, protected_target :: p, a(:)
    type(initialized), pointer, protected_target :: b
    integer, pointer :: ordinary
    allocate(p, source=1)
    allocate(a(2), source=1)
    allocate(b, source=initialized())
    !ERROR: ALLOCATE of a PROTECTED_TARGET pointer requires SOURCE=
    allocate(p)
    !ERROR: ALLOCATE of a PROTECTED_TARGET pointer requires SOURCE=
    allocate(p, mold=1)
    ! Default initialization does not replace the SOURCE= requirement.
    !ERROR: ALLOCATE of a PROTECTED_TARGET pointer requires SOURCE=
    allocate(b)
    !ERROR: A PROTECTED_TARGET pointer may not appear in DEALLOCATE
    deallocate(p)
    !ERROR: A PROTECTED_TARGET pointer may not appear in DEALLOCATE
    deallocate(a)
    p => null()
    nullify(a, b)
    allocate(ordinary)
    ordinary = 1
    deallocate(ordinary)
  end subroutine

  subroutine associations(p, target)
    integer, pointer, protected_target, intent(in) :: p
    integer, target, intent(in) :: target
    integer, pointer, protected_target :: q
    logical :: status
    q => p
    q => target
    status = associated(q, p)
    q => null(q)
    !ERROR: 'p' may not appear in NULLIFY
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    nullify(p)
  end subroutine
end module
