! RUN: %python %S/test_errors.py %s %flang_fc1
! The target of a PROTECTED_TARGET pointer, and any subobject of that target,
! is not definable via that pointer (F'2028 C866, C868, C869, 8.5.16p2).

module m
  type :: t
    integer :: n
    integer :: a(2)
    integer, allocatable :: al
    integer, pointer :: next
    procedure(), nopass, pointer :: pp
  end type
  type :: wrap
    integer, pointer, protected_target :: q
  end type
  interface assignment(=)
    module procedure assign_int
  end interface
 contains
  subroutine assign_int(a, b)
    type(t), intent(in out) :: a
    integer, intent(in) :: b
  end
  function get() result(r)
    integer, pointer, protected_target :: r
    allocate(r, source=1)
  end
end

subroutine definitions(p, pa, pt, pc, pz, w, x, y)
  use m
  integer, pointer, protected_target :: p, pa(:)
  type(t), pointer, protected_target :: pt
  character(4), pointer, protected_target :: pc
  complex, pointer, protected_target :: pz
  type(wrap) :: w
  integer, target :: x
  integer, pointer :: y
  integer :: j
  external :: ext
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  p = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  if (.false.) p = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'pa' has the PROTECTED_TARGET attribute
  pa(1) = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'pa' has the PROTECTED_TARGET attribute
  pa(1:2) = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  pt%n = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  pt%a(2) = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  pt%al = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'pc' has the PROTECTED_TARGET attribute
  pc(1:2) = 'ab'
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'pz' has the PROTECTED_TARGET attribute
  pz%re = 1.
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'q' has the PROTECTED_TARGET attribute
  w%q = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'r' has the PROTECTED_TARGET attribute
  get() = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  pt = 1
  !ERROR: Input variable 'p' is not definable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  read *, p
  !ERROR: IOSTAT variable 'p' is not definable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  read(*, *, iostat=p) j
  !ERROR: Internal file variable 'pc' is not definable
  !BECAUSE: 'pc' has the PROTECTED_TARGET attribute
  write(pc, '(i4)') j
  !ERROR: 'p' may not be used as a DO variable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  do p = 1, 2
  end do
  !ERROR: 'p' has the PROTECTED_TARGET attribute
  print *, (j, p = 1, 2)
  associate (a => p, b => pt%a)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'p' has the PROTECTED_TARGET attribute
    a = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
    b(1) = 1
  end associate
  !ERROR: Name in ALLOCATE statement is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  allocate(pt%al)
  !ERROR: Name in DEALLOCATE statement is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  deallocate(pt%al)
  ! A pointer component in the target is part of the target.
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  pt%next => x
  !ERROR: 'next' may not appear in NULLIFY
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  nullify(pt%next)
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  pt%pp => ext
  !ERROR: Name in ALLOCATE statement is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  allocate(pt%next)
  !ERROR: Name in DEALLOCATE statement is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  deallocate(pt%next)
  ! The target of that component is not part of the target, so it may be
  ! defined.
  pt%next = 1
  ! The association of a PROTECTED_TARGET pointer may change.
  p => x
  p => y
  nullify(p)
  pa => null()
  w%q => x
  ! The target can still be defined other than via the pointer.
  x = 2
  y => x
  y = 3
  print *, p, pa, pt%n, pc, pz, w%q, get()
end

subroutine allocations(p, pa, w)
  use m
  integer, pointer, protected_target :: p, pa(:)
  type(wrap) :: w
  !ERROR: An ALLOCATE statement with PROTECTED_TARGET pointer 'p' must have SOURCE=
  allocate(p)
  !ERROR: An ALLOCATE statement with PROTECTED_TARGET pointer 'pa' must have SOURCE=
  allocate(pa(2))
  !ERROR: An ALLOCATE statement with PROTECTED_TARGET pointer 'pa' must have SOURCE=
  allocate(pa, mold=[1, 2])
  !ERROR: An ALLOCATE statement with PROTECTED_TARGET pointer 'q' must have SOURCE=
  allocate(w%q)
  allocate(p, source=1)
  allocate(pa, source=[1, 2])
  allocate(w%q, source=p)
  !ERROR: Object in DEALLOCATE statement is not deallocatable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  deallocate(p)
  !ERROR: Object in DEALLOCATE statement is not deallocatable
  !BECAUSE: 'q' has the PROTECTED_TARGET attribute
  deallocate(w%q)
end

subroutine intent_in(p)
  integer, pointer, protected_target, intent(in) :: p
  integer, target :: x
  print *, p
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'p' is an INTENT(IN) dummy argument
  p => x
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  p = 1
end

subroutine select(up)
  class(*), pointer, protected_target :: up
  select type (a => up)
  type is (integer)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'up' has the PROTECTED_TARGET attribute
    a = 1
  end select
end

subroutine select_rank(ar, x)
  integer, pointer, protected_target :: ar(..)
  integer, target :: x(2)
  ! In a variable definition context, an associate name stands for its
  ! selector (F'2028 20.6.7(11)).
  select rank (r => ar)
  rank (0)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
    r = 1
    !ERROR: Input variable 'r' is not definable
    !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
    read *, r
  rank (1)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
    r(1) = 1
    !ERROR: An ALLOCATE statement with PROTECTED_TARGET pointer 'ar' must have SOURCE=
    allocate(r(2))
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
    deallocate(r)
    ! Accepted: F'2028 11.1.3.3p5 forbids these pointer association contexts
    ! for the associate name, but not as a constraint.
    allocate(r, source=x)
    nullify(r)
    r => x
  rank default
    nullify(r)
    select rank (s => r)
    rank (2)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
      s(1, 1) = 1
    end select
  end select
end

subroutine namelist_read(p)
  integer, pointer, protected_target :: p
  namelist /nml/ p
  !ERROR: NAMELIST input group must not contain undefinable item 'p'
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  read(*, nml=nml)
  write(*, nml=nml)
end
