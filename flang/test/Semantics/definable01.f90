! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not="No intrinsic or user-defined ASSIGNMENT(=)"
! Test WhyNotDefinable() explanations

module prot
  real, protected :: prot
  type :: ptype
    real, pointer :: ptr
    real :: x
  end type
  type(ptype), protected :: protptr
 contains
  subroutine ok
    prot = 0. ! ok
  end subroutine
end module

module m
  use iso_fortran_env
  use prot
  type :: t1
    type(lock_type) :: lock
  end type
  type :: t2
    type(t1) :: x1
    real :: x2
  end type
  type(t2) :: t2static
  type list
    real a
    type(list), pointer :: prev, next
  end type
  character(*), parameter :: internal = '0'
 contains
  subroutine test1(dummy)
    real :: arr(2)
    integer, parameter :: j3 = 666
    type(ptype), intent(in) :: dummy
    type(t2) :: t2var
    associate (a => 3+4)
      !CHECK: error: Input variable 'a' is not definable
      !CHECK: because: 'a' is construct associated with an expression
      read(internal,*) a
    end associate
    associate (a => arr([1])) ! vector subscript
      !CHECK: error: Input variable 'a' is not definable
      !CHECK: because: Construct association 'a' has a vector subscript
      read(internal,*) a
    end associate
    associate (a => arr(2:1:-1))
      read(internal,*) a ! ok
    end associate
    !CHECK: error: Input variable 'j3' is not definable
    !CHECK: because: '666_4' is not a variable
    read(internal,*) j3
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 't2var' is an entity with either an EVENT_TYPE or LOCK_TYPE
    t2var = t2static
    t2var%x2 = 0. ! ok
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 'prot' is protected in this scope
    prot = 0.
    protptr%ptr = 0. ! ok
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 'dummy' is an INTENT(IN) dummy argument
    dummy%x = 0.
    dummy%ptr = 0. ! ok
  end subroutine
  pure subroutine test2(ptr)
    integer, pointer, intent(in) :: ptr
    !CHECK: error: Input variable 'ptr' is not definable
    !CHECK: because: 'ptr' is externally visible via 'ptr' and not definable in a pure subprogram
    read(internal,*) ptr
  end subroutine
  subroutine test3(objp, procp)
    real, intent(in), pointer :: objp
    procedure(sin), pointer, intent(in) :: procp
    !CHECK: error: Actual argument associated with INTENT(IN OUT) dummy argument 'op=' is not definable
    !CHECK: because: 'objp' is an INTENT(IN) dummy argument
    call test3a(objp)
    !CHECK: error: Actual argument associated with INTENT(IN OUT) procedure pointer dummy argument 'pp=' is not definable
    !CHECK: because: 'procp' is an INTENT(IN) dummy argument
    call test3b(procp)
  end subroutine
  subroutine test3a(op)
    real, intent(in out), pointer :: op
  end subroutine
  subroutine test3b(pp)
    procedure(sin), pointer, intent(in out) :: pp
  end subroutine
  subroutine test4(p)
    type(ptype), pointer, intent(in) :: p
    p%x = 1.
    p%ptr = 1. ! ok
    nullify(p%ptr) ! ok
    !CHECK: error: 'p' may not appear in NULLIFY
    !CHECK: because: 'p' is an INTENT(IN) dummy argument
    nullify(p)
  end
  subroutine test5(np)
    type(ptype), intent(in) :: np
    !CHECK: error: 'ptr' may not appear in NULLIFY
    !CHECK: because: 'np' is an INTENT(IN) dummy argument
    nullify(np%ptr)
  end
  pure function test6(lp)
    type(list), pointer :: lp
    !CHECK: error: The left-hand side of a pointer assignment is not definable
    !CHECK: because: 'lp' may not be defined in pure subprogram 'test6' because it is a POINTER dummy argument of a pure function
    lp%next%next => null()
  end
  pure subroutine test7(lp)
    type(list), pointer :: lp
    lp%next%next => null() ! ok
  end
end module
program main
  use iso_fortran_env, only: lock_type
  type(lock_type) lock
  interface
    subroutine inlock(lock)
      import lock_type
      type(lock_type), intent(in) :: lock
    end
    subroutine outlock(lock)
      import lock_type
      !CHECK: error: An INTENT(OUT) dummy argument may not be, or contain, EVENT_TYPE or LOCK_TYPE
      type(lock_type), intent(out) :: lock
    end
    subroutine inoutlock(lock)
      import lock_type
      type(lock_type), intent(in out) :: lock
    end
  end interface
  call inlock(lock) ! ok
  call inoutlock(lock) ! ok
  !CHECK: error: Actual argument associated with INTENT(OUT) dummy argument 'lock=' is not definable
  call outlock(lock)
end

module protected_target_definition_contexts
  integer, target :: target
  type box
    integer :: value
    integer, pointer :: link
  end type
contains
  function get() result(r)
    integer, pointer, protected_target :: r
    r => target
  end function
  subroutine definitions(p, chars, logical_pointer)
    integer, pointer, protected_target :: p
    character(40), pointer, protected_target :: chars
    logical, pointer, protected_target :: logical_pointer
    integer, allocatable :: allocated
    integer :: status
    character(40) :: input
    namelist /readonly_namelist/ p
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    if (.false.) p = 1
    !CHECK: error: Input variable 'p' is not definable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    read(*,*) p
    !CHECK: error: 'p' may not be used as a DO variable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    do p = 1, 2
    end do
    !CHECK: error: NAMELIST input group must not contain undefinable item 'p'
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    read(*, nml=readonly_namelist)
    !CHECK: error: Internal file variable 'chars' is not definable
    !CHECK: because: 'chars' is a PROTECTED_TARGET pointer
    write(chars,*) 1
    !CHECK: error: NEWUNIT variable 'p' is not definable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    open(file='unused', newunit=p)
    !CHECK: error: IOSTAT variable 'p' is not definable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    read(1,*,iostat=p)
    !CHECK: error: IOMSG variable 'chars' is not definable
    !CHECK: because: 'chars' is a PROTECTED_TARGET pointer
    read(1,*,iostat=status,iomsg=chars)
    !CHECK: error: SIZE variable 'p' is not definable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    read(1,'(A)',advance='no',size=p) input
    !CHECK: error: EXIST variable 'logical_pointer' is not definable
    !CHECK: because: 'logical_pointer' is a PROTECTED_TARGET pointer
    inquire(unit=1,exist=logical_pointer)
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 'chars' is a PROTECTED_TARGET pointer
    chars(2:4) = 'abc'
    !CHECK: error: Input variable 'chars' is not definable
    !CHECK: because: 'chars' is a PROTECTED_TARGET pointer
    read(*,*) chars(2:4)
    !CHECK: error: STAT variable 'p' is not definable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    allocate(allocated,stat=p)
    !CHECK: error: ERRMSG variable 'chars' is not definable
    !CHECK: because: 'chars' is a PROTECTED_TARGET pointer
    deallocate(allocated,errmsg=chars)
  end subroutine
  subroutine function_results
    print *, get()
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 'r' is a PROTECTED_TARGET pointer
    get() = 1
    !CHECK: error: Input variable 'get()' is not definable
    !CHECK: because: 'r' is a PROTECTED_TARGET pointer
    read(*,*) get()
    associate(a => get())
      !CHECK: error: Left-hand side of assignment is not definable
      !CHECK: because: 'r' is a PROTECTED_TARGET pointer
      a = 1
    end associate
  end subroutine
  subroutine iolength(p, array, object, input)
    integer, pointer, protected_target :: p, array(:)
    type(box), pointer, protected_target :: object
    integer, intent(in) :: input
    integer, pointer :: ordinary
    integer :: length
    !CHECK: error: IOLENGTH variable 'p' is not definable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    inquire(iolength=p) 1
    !CHECK: error: IOLENGTH variable 'array' is not definable
    !CHECK: because: 'array' is a PROTECTED_TARGET pointer
    inquire(iolength=array(1)) 1
    !CHECK: error: IOLENGTH variable 'object' is not definable
    !CHECK: because: 'object' is a PROTECTED_TARGET pointer
    inquire(iolength=object%value) 1
    associate(alias => p)
      !CHECK: error: IOLENGTH variable 'alias' is not definable
      !CHECK: because: 'p' is a PROTECTED_TARGET pointer
      inquire(iolength=alias) 1
    end associate
    !CHECK: error: IOLENGTH variable 'get()' is not definable
    !CHECK: because: 'r' is a PROTECTED_TARGET pointer
    inquire(iolength=get()) 1
    inquire(iolength=object%link) p
    inquire(iolength=length) p
    inquire(iolength=ordinary) p
    ! Preserve the existing INTENT(IN) IOLENGTH behavior.
    inquire(iolength=input) p
  end subroutine
end module

module protected_target_defined_assignment
  type t
    integer :: value
  contains
    procedure :: assign
    generic :: assignment(=) => assign
  end type
contains
  subroutine assign(lhs, rhs)
    class(t), intent(out) :: lhs
    integer, intent(in) :: rhs
    lhs%value = rhs
  end subroutine
  subroutine definitions(p, ordinary)
    type(t), pointer, protected_target :: p
    type(t), pointer :: ordinary
    !CHECK: error: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'lhs='
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 'p' is a PROTECTED_TARGET pointer
    p = 1
    ordinary = 1
  end subroutine
end module
