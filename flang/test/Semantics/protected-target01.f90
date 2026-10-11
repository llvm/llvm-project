! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine declarations
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  integer, protected_target :: not_pointer
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  integer, allocatable, protected_target :: not_pointer_either
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  integer :: statement_not_pointer
  protected_target :: statement_not_pointer
  protected_target ordered
  integer :: ordered
  pointer :: ordered
  type bad_component
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    integer, protected_target :: c
  end type
  interface
    subroutine proc_interface
    end subroutine
  end interface
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  interface generic
    subroutine specific
    end subroutine
  end interface
  protected_target generic
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  procedure(proc_interface), pointer :: proc
  protected_target proc
  !ERROR: A PROTECTED_TARGET pointer may not be in a common block
  integer, pointer, protected_target :: common_pointer
  common /blk/ common_pointer
end

subroutine common_components
  ! x, not its pointer component, is in COMMON (F2028 8.11.3.1p3).
  type inner
    sequence
    integer, pointer, protected_target :: p
  end type
  type outer
    sequence
    type(inner) :: c
  end type
  type(outer) :: x
  type(outer), pointer :: ordinary
  common /components/ x
  common /pointers/ ordinary
end

subroutine allocation
  type t
    integer, pointer, protected_target :: c
  end type
  type(t) :: x
  integer, pointer, protected_target :: p, q(:)
  integer, pointer :: ordinary
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(p)
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(p, mold=1)
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(integer :: p)
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(x%c)
  allocate(p, x%c, source=1)
  allocate(q, source=[1, 2])
  allocate(ordinary)
  !ERROR: A PROTECTED_TARGET pointer may not be deallocated
  deallocate(p)
  !ERROR: A PROTECTED_TARGET pointer may not be deallocated
  deallocate(x%c)
  deallocate(ordinary)
  nullify(p, q, x%c)
end

module definitions
  type t
    integer :: value, array(2)
    character(4) :: text
    complex :: z
    integer, allocatable :: a
    integer, pointer :: link
    integer, pointer, protected_target :: ro
    procedure(), pointer, nopass :: proc
  end type
contains
  subroutine scalar_and_sections
    integer, target :: x, y(2)
    integer, pointer, protected_target :: p, q(:)
    character(4), target :: text
    character(:), pointer, protected_target :: chars
    complex, target :: z
    complex, pointer, protected_target :: c
    p => x
    q => y
    chars => text
    c => z
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    if (.false.) p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'q' is not definable
    q(1:2) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'q' is not definable
    q([1, 2]) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'chars' is not definable
    chars(1:2) = 'ab'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'c' is not definable
    c%re = 1
    !ERROR: Input variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    read *, p
    print *, x
    !ERROR: IOSTAT variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    read (*, *, iostat=p) x
    y(p) = 1
    x = 2
    print *, p, q, chars, c, associated(p), size(q), len(chars)
    p => y(1)
    nullify(p, q, chars, c)
  end

  subroutine components_and_associations
    type(t), target :: x
    type(t), pointer, protected_target :: p
    integer, target :: target
    p => x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%value = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%array(1) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%text(1:1) = 'a'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%z%im = 1
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%link => target
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%proc => callee
    !ERROR: 'proc' may not appear in NULLIFY
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(p%proc)
    call p%proc()
    x%proc => callee
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(p%a, source=1)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    deallocate(p%a)
    ! The target of an ordinary pointer component is not a subobject of p.
    p%link = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'ro' is not definable
    x%ro = 1
    x%ro => target
    associate(a => p%array)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a(1) = 1
    end associate
    associate(a => p)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a%value = 1
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a%link => target
      a%link = 1
    end associate
    nullify(p)
  end

  subroutine callee
  end

  subroutine association_status(p, x, ordinary)
    integer, pointer, protected_target, intent(in) :: p
    integer, target, intent(in) :: x
    integer, pointer, intent(in) :: ordinary
    integer, pointer, protected_target :: q
    q => x
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    p => x
    !ERROR: 'p' may not appear in NULLIFY
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    nullify(p)
    ordinary = 1
    nullify(q)
  end

  subroutine definition_contexts
    integer, pointer, protected_target :: p
    character(16), pointer, protected_target :: text
    logical, pointer, protected_target :: exists
    integer, allocatable :: a
    integer :: x
    namelist /group/ p
    !ERROR: 'p' may not be used as a DO variable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    do p = 1, 2
    end do
    !ERROR: The target of PROTECTED_TARGET pointer 'p' is not definable
    print *, (x, p = 1, 2)
    !ERROR: NAMELIST input group must not contain undefinable item 'p'
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    read (*, nml=group)
    write (*, nml=group)
    !ERROR: Internal file variable 'text' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'text' is not definable
    write (text, *) x
    !ERROR: NEWUNIT variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    open (newunit=p, status='scratch')
    !ERROR: EXIST variable 'exists' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'exists' is not definable
    inquire (file='file', exist=exists)
    !ERROR: IOMSG variable 'text' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'text' is not definable
    read (*, *, iostat=x, iomsg=text) x
    !ERROR: STAT variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(a, stat=p)
    !ERROR: ERRMSG variable 'text' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'text' is not definable
    deallocate(a, stat=x, errmsg=text)
  end

  subroutine type_guard(p)
    class(t), pointer, protected_target :: p
    select type (alias => p)
    type is (t)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      alias%value = 1
      print *, alias%value
    end select
  end

  subroutine rank_guard(p, q)
    integer, pointer, protected_target :: p(..)
    integer, pointer, protected_target :: q(:)
    select rank (alias => p)
    rank (1)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      alias(1) = 1
      print *, alias
      nullify(alias)
      alias => q
      !ERROR: A PROTECTED_TARGET pointer may not be deallocated
      deallocate(alias)
      !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
      allocate(alias(2))
    rank default
      nullify(alias)
    end select
  end
end module
