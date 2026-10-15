! RUN: %python %S/test_errors.py %s %flang_fc1
! F'2028 C866, C868, C869 and 8.5.16p2: target definition differs from
! pointer association definition, including through nested components.

module protected_target_definitions
  type leaf
    integer :: value
    integer, pointer :: writable
    integer, pointer, protected_target :: readonly
    integer, allocatable :: alloc
    procedure(), pointer, nopass :: proc
  end type
  type node
    type(leaf) :: child
  end type
  type holder
    type(node), pointer, protected_target :: p
  end type
contains
  subroutine values(p, a, c, z)
    integer, pointer, protected_target :: p, a(:)
    character(8), pointer, protected_target :: c
    complex, pointer, protected_target :: z
    character(8) :: input = '1'
    integer, allocatable :: x
    integer :: status
    integer :: ordinary(10)
    ordinary(p) = 1 ! Reading a protected pointer in a subscript is permitted.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    if (.false.) p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'a' is not definable
    a(:) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'a' is not definable
    a(1) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'c' is not definable
    c(2:4) = 'abc'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'z' is not definable
    z%re = 1.
    !ERROR: Input variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    read(input, *) p
    !ERROR: IOLENGTH variable 'a' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'a' is not definable
    inquire(iolength=a(1)) status
    inquire(iolength=ordinary(p)) status
    !ERROR: 'p' may not be used as a DO variable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    do p = 1, 10
    end do
    !ERROR: The target of PROTECTED_TARGET pointer 'p' is not definable
    print *, (1, p = 1, 10)
    !ERROR: IOSTAT variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    read(input, *, iostat=p) x
    !ERROR: STAT variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(x, stat=p)
    !ERROR: ERRMSG variable 'c' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'c' is not definable
    allocate(x, stat=status, errmsg=c)
    !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
    allocate(p)
    !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
    allocate(p, mold=1)
    allocate(p, source=1)
    allocate(a(3), source=1)
    !ERROR: A PROTECTED_TARGET pointer may not be deallocated
    deallocate(p)
    !ERROR: A PROTECTED_TARGET pointer may not be deallocated
    deallocate(a)
    nullify(p, a)
    p => null()
    print *, c, z
    associate (alias => p)
      call pointer_in(alias)
      !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      call value_inout(alias)
    end associate
  end subroutine

  subroutine definition_contexts(p, c, u, q, n)
    integer, pointer, protected_target :: p, u, q
    character(8), pointer, protected_target :: c
    integer, intent(in) :: n
    integer :: i
    namelist /g/ p
    !ERROR: NAMELIST input group must not contain undefinable item 'p'
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    read(10, nml=g)
    !ERROR: Internal file variable 'c' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'c' is not definable
    write(c, '(a)') 'x'
    !ERROR: NEWUNIT variable 'u' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'u' is not definable
    open(newunit=u, status='scratch')
    !ERROR: IOLENGTH variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    inquire(iolength=p) i
    ! Keep the existing IOLENGTH= behavior for INTENT(IN) variables.
    inquire(iolength=n) i
    !ERROR: 'q' may not appear in a locality-spec because it is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'q' is not definable
    do concurrent (i=1:3) local(q)
    end do
  end subroutine

  subroutine nested(p, x, h)
    type(node), pointer, protected_target :: p
    type(node), target :: x
    type(holder) :: h
    integer, target :: target
    p => x
    h%p => x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%child%value = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    h%p%child%value = 1
    !ERROR: IOLENGTH variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    inquire(iolength=p%child%value) target
    !ERROR: IOLENGTH variable 'h' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    inquire(iolength=h%p%child%value) target
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'readonly' is not definable
    x%child%readonly = 1
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%child%writable => target
    !ERROR: 'writable' may not appear in NULLIFY
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(p%child%writable)
    !ERROR: 'proc' may not appear in NULLIFY
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(p%child%proc)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(p%child%alloc, source=1)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    deallocate(p%child%writable)
    !ERROR: A PROTECTED_TARGET pointer may not be deallocated
    deallocate(x%child%readonly)
    !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
    allocate(x%child%readonly)
    allocate(x%child%readonly, source=1)
    allocate(h%p, source=x)
    nullify(h%p)
    p%child%writable = 1 ! Defines another pointer's target, not a subobject of p.
    inquire(iolength=p%child%writable) target
    associate (a => p%child%value)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
      !ERROR: IOLENGTH variable 'a' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      inquire(iolength=a) target
    end associate
    associate (a => x%child%readonly)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'readonly' is not definable
      a = 1
    end associate
  end subroutine

  subroutine intent_in(p)
    integer, pointer, protected_target, intent(in) :: p
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    p => null()
    !ERROR: 'p' may not appear in NULLIFY
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    nullify(p)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    allocate(p, source=1)
    print *, p
  end subroutine

  subroutine coarray_definitions(p, l, team, n, acquired, input_team, ev, nv)
    use iso_fortran_env, only: event_type, lock_type, notify_type, team_type
    integer, pointer, protected_target :: p
    logical, pointer, protected_target :: l
    type(team_type), pointer, protected_target :: team
    integer, intent(in) :: n
    logical, intent(in) :: acquired
    type(team_type), intent(in) :: input_team
    type(event_type), intent(in) :: ev[*]
    type(notify_type), intent(in) :: nv[*]
    type state
      integer :: status
      logical :: acquired
      type(team_type) :: team
    end type
    type(state), pointer, protected_target :: s
    type protected_pointers
      type(event_type), pointer, protected_target :: event
      type(notify_type), pointer, protected_target :: notify
    end type
    type(protected_pointers) :: h
    type(lock_type), save :: lock_var[*]
    integer, save :: coarray[*]
    integer :: x
    !ERROR: Image selector STAT variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    x = coarray[1, stat=p]
    !ERROR: Image selector STAT variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 's' is not definable
    coarray[1, stat=s%status] = x
    !ERROR: NOTIFY variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'notify' is not definable
    coarray[1, notify=h%notify] = x
    !ERROR: Event variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'event' is not definable
    event post(h%event)
    !ERROR: Event variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'event' is not definable
    event wait(h%event)
    !ERROR: ACQUIRED_LOCK variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'l' is not definable
    lock(lock_var, acquired_lock=l)
    !ERROR: ACQUIRED_LOCK variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 's' is not definable
    lock(lock_var, acquired_lock=s%acquired)
    !ERROR: Team variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'team' is not definable
    form team(1, team)
    !ERROR: Team variable is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 's' is not definable
    form team(1, s%team)
    associate (alias => s%team)
      !ERROR: Team variable is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 's' is not definable
      form team(1, alias)
    end associate
    ! Keep the existing INTENT(IN) behavior in these contexts.
    x = coarray[1, stat=n]
    coarray[1, notify=nv] = x
    event post(ev)
    event wait(ev)
    lock(lock_var, acquired_lock=acquired)
    form team(1, input_team)
  end subroutine

  subroutine pointer_in(x)
    integer, pointer, intent(in) :: x
  end subroutine
  subroutine value_inout(x)
    integer, intent(inout) :: x
  end subroutine
end module
