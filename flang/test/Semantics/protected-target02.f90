! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 draft J3/26-007r1 C866, C868, C869 and 8.5.16 paragraph 2.

module definitions
  implicit none
  abstract interface
    subroutine callback()
    end subroutine
  end interface
  type node
    integer :: value
    integer, pointer :: writable
    integer, pointer, protected_target :: readonly
    integer, allocatable :: alloc
    procedure(callback), pointer, nopass :: proc
  end type
  type container
    type(node), pointer, protected_target :: link
  end type
  type initialized
    integer :: value = 1
  end type
contains
  subroutine value_contexts(p, a, c, z, l)
    integer, pointer, protected_target :: p, a(:)
    character(8), pointer, protected_target :: c
    complex, pointer, protected_target :: z
    logical, pointer, protected_target :: l
    type(node), pointer, protected_target :: object
    integer :: x
    integer, pointer :: ordinary
    namelist /nml/ p
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    if (.false.) p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'a' is not definable via that pointer
    a(1:2) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'c' is not definable via that pointer
    c(2:3) = "xy"
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'z' is not definable via that pointer
    z%im = 1.
    !ERROR: Input variable 'p' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    read(*,*) p
    !ERROR: Internal file variable 'c' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'c' is not definable via that pointer
    write(c,*) 1
    !ERROR: IOSTAT variable 'p' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    read(*,*,iostat=p) x
    !ERROR: NEWUNIT variable 'p' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    open(newunit=p,status="scratch")
    !ERROR: SIZE variable 'p' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    inquire(unit=5,size=p)
    !ERROR: OPENED variable 'l' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'l' is not definable via that pointer
    inquire(unit=5,opened=l)
    !ERROR: IOLENGTH variable 'p' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    inquire(iolength=p) 1
    !ERROR: IOLENGTH variable 'object' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable via that pointer
    inquire(iolength=object%value) 1
    !ERROR: IOLENGTH variable 'a' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'a' is not definable via that pointer
    inquire(iolength=a(1)) 1
    inquire(iolength=ordinary) 1
    inquire(iolength=x) 1
    !ERROR: STAT variable 'p' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    allocate(ordinary,stat=p)
    !ERROR: ERRMSG variable 'c' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'c' is not definable via that pointer
    deallocate(ordinary,errmsg=c)
    !ERROR: NAMELIST input group must not contain undefinable item 'p'
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    read(*,nml=nml)
    !ERROR: 'p' may not be used as a DO variable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    do p = 1, 2
    end do
    !ERROR: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    read(*,*) (x, p=1,2)
    !ERROR: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    print *, (x, p=1,2)
    print *, p, a, c, z, l
  end subroutine

  subroutine allocation(p, a, object)
    integer, pointer, protected_target :: p, a(:)
    type(initialized), pointer, protected_target :: default_initialized
    type(node) :: object
    !ERROR: PROTECTED_TARGET pointer 'p' in ALLOCATE requires a SOURCE= clause
    allocate(p)
    !ERROR: PROTECTED_TARGET pointer 'p' in ALLOCATE requires a SOURCE= clause
    allocate(p,mold=1)
    !ERROR: PROTECTED_TARGET pointer 'readonly' in ALLOCATE requires a SOURCE= clause
    allocate(object%readonly)
    !ERROR: PROTECTED_TARGET pointer 'default_initialized' in ALLOCATE requires a SOURCE= clause
    allocate(default_initialized)
    allocate(p,source=42)
    allocate(a(2),source=42)
    allocate(object%readonly,source=42)
    allocate(default_initialized,source=initialized())
    !ERROR: PROTECTED_TARGET pointer 'p' may not appear in DEALLOCATE
    deallocate(p)
    !ERROR: PROTECTED_TARGET pointer 'readonly' may not appear in DEALLOCATE
    deallocate(object%readonly)
    nullify(p, a, object%readonly)
  end subroutine

  subroutine independent_restriction(p)
    integer, pointer, protected_target, intent(in) :: p
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    allocate(p,source=1)
    !ERROR: 'p' may not appear in NULLIFY
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    nullify(p)
  end subroutine

  subroutine subobjects(p, root, x)
    type(node), pointer, protected_target :: p
    type(container) :: root
    type(node), target :: x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p%value = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'link' is not definable via that pointer
    root%link%value = 1
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    allocate(p%writable,source=1)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    allocate(p%alloc,source=1)
    !ERROR: 'writable' may not appear in NULLIFY
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    nullify(p%writable)
    !ERROR: 'proc' may not appear in NULLIFY
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    nullify(p%proc)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p%writable => x%value
    p%writable = 1 ! A different pointer's target, not a subobject of p's target.
    x%value = 1
    p => x
    nullify(p)
  end subroutine

  subroutine associations(p, a, poly, rank_pointer)
    integer, pointer, protected_target :: p, a(:), rank_pointer(..)
    class(node), pointer, protected_target :: poly
    associate (v => p, section => a(1:2))
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      v = 1
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'a' is not definable via that pointer
      section(1) = 1
    end associate
    select type (v => poly)
    type is (node)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'poly' is not definable via that pointer
      v%value = 1
    end select
    select rank (v => rank_pointer)
    rank (1)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'rank_pointer' is not definable via that pointer
      v(1) = 1
      !ERROR: PROTECTED_TARGET pointer 'v' in ALLOCATE requires a SOURCE= clause
      allocate(v(2))
      !ERROR: PROTECTED_TARGET pointer 'v' in ALLOCATE requires a SOURCE= clause
      allocate(v(2),mold=[1,2])
      allocate(v(2),source=1)
      !ERROR: PROTECTED_TARGET pointer 'v' may not appear in DEALLOCATE
      deallocate(v)
      nullify(v)
    rank default
      !ERROR: PROTECTED_TARGET pointer 'v' may not appear in DEALLOCATE
      deallocate(v)
      nullify(v)
      select rank (scalar => v)
      rank (0)
        !ERROR: PROTECTED_TARGET pointer 'scalar' in ALLOCATE requires a SOURCE= clause
        allocate(scalar)
        allocate(scalar,source=1)
        !ERROR: PROTECTED_TARGET pointer 'scalar' may not appear in DEALLOCATE
        deallocate(scalar)
        nullify(scalar)
      end select
    end select
  end subroutine

  subroutine rank_allocation(p, ordinary)
    integer, pointer, protected_target :: p(..)
    integer, pointer :: ordinary(..)
    select rank (p)
    rank (0)
      !ERROR: PROTECTED_TARGET pointer 'p' in ALLOCATE requires a SOURCE= clause
      allocate(p)
      !ERROR: PROTECTED_TARGET pointer 'p' in ALLOCATE requires a SOURCE= clause
      allocate(p,mold=1)
      allocate(p,source=1)
      !ERROR: PROTECTED_TARGET pointer 'p' may not appear in DEALLOCATE
      deallocate(p)
      nullify(p)
    rank (1)
      !ERROR: PROTECTED_TARGET pointer 'p' in ALLOCATE requires a SOURCE= clause
      allocate(p(2))
      !ERROR: PROTECTED_TARGET pointer 'p' in ALLOCATE requires a SOURCE= clause
      allocate(p(2),mold=[1,2])
      allocate(p(2),source=1)
      !ERROR: PROTECTED_TARGET pointer 'p' may not appear in DEALLOCATE
      deallocate(p)
      nullify(p)
    rank default
      !ERROR: PROTECTED_TARGET pointer 'p' may not appear in DEALLOCATE
      deallocate(p)
      nullify(p)
    end select
    select rank (v => ordinary)
    rank (0)
      allocate(v)
      allocate(v,mold=1)
      allocate(v,source=1)
      deallocate(v)
      nullify(v)
    rank (1)
      allocate(v(2))
      allocate(v(2),mold=[1,2])
      allocate(v(2),source=1)
      deallocate(v)
      nullify(v)
    rank default
      deallocate(v)
      nullify(v)
    end select
  end subroutine

  subroutine image_contexts(team, acquired)
    use iso_fortran_env, only: team_type, event_type, lock_type
    type(team_type), pointer, protected_target :: team
    logical, pointer, protected_target :: acquired
    type image_holder
      type(event_type), pointer, protected_target :: event
      type(lock_type), pointer, protected_target :: lock_variable
    end type
    type ordinary_holder
      type(event_type), pointer :: event
      type(lock_type), pointer :: lock_variable
    end type
    type(image_holder) :: h
    type(ordinary_holder) :: ordinary
    type(event_type), target, save :: event_target[*]
    type(lock_type), target, save :: lock_target[*]
    h%event => event_target
    h%lock_variable => lock_target
    ordinary%event => event_target
    ordinary%lock_variable => lock_target
    !ERROR: Team variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'team' is not definable via that pointer
    form team (1, team)
    !ERROR: Event variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'event' is not definable via that pointer
    event post(h%event)
    !ERROR: Event variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'event' is not definable via that pointer
    event wait(h%event)
    !ERROR: Lock variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'lock_variable' is not definable via that pointer
    lock(h%lock_variable)
    !ERROR: Lock variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'lock_variable' is not definable via that pointer
    unlock(h%lock_variable)
    !ERROR: ACQUIRED_LOCK= variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'acquired' is not definable via that pointer
    lock(lock_target,acquired_lock=acquired)
    event post(ordinary%event)
    event wait(ordinary%event)
    lock(ordinary%lock_variable)
    unlock(ordinary%lock_variable)
  end subroutine

  subroutine image_selectors(co, p, a, object, root, ordinary)
    use iso_fortran_env, only: notify_type
    integer :: co[*], value
    integer, pointer, protected_target :: p, a(:)
    type(node), pointer, protected_target :: object
    type(container) :: root
    integer, pointer :: ordinary
    type holder
      type(notify_type), pointer, protected_target :: notify
    end type
    type nested_holder
      type(holder) :: member
    end type
    type ordinary_holder
      type(notify_type), pointer :: notify
    end type
    type(holder) :: h
    type(nested_holder) :: nested
    type(ordinary_holder) :: writable
    type(notify_type), target, save :: notify_target[*]
    h%notify => notify_target
    nested%member%notify => notify_target
    writable%notify => notify_target
    !ERROR: Image selector STAT variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    co[1,stat=p] = 1
    !ERROR: Image selector STAT variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    value = co[1,stat=p]
    !ERROR: Image selector STAT variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'a' is not definable via that pointer
    co[1,stat=a(1)] = 1
    !ERROR: Image selector STAT variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable via that pointer
    co[1,stat=object%value] = 1
    !ERROR: Image selector STAT variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'link' is not definable via that pointer
    value = co[1,stat=root%link%value]
    !ERROR: Image selector NOTIFY variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'notify' is not definable via that pointer
    co[1,notify=h%notify] = 1
    !ERROR: Image selector NOTIFY variable is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'notify' is not definable via that pointer
    co[1,notify=nested%member%notify] = 1
    co[1,stat=ordinary] = 1
    value = co[1,stat=value]
    co[1,stat=object%writable] = 1
    co[1,notify=writable%notify] = 1
  end subroutine
end module
