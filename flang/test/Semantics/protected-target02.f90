! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 PROTECTED_TARGET: the target of the pointer is not definable through
! it (J3/26-007r1 8.5.16 paragraph 2, C866, C868, C869)
module m
  type t
    real :: x
    real, pointer :: q
    real, pointer, protected_target :: c
    real, allocatable :: a
  end type
  type tpp
    procedure(f0), pointer, nopass :: pp
  end type
  real, target :: tgt, tgtarr(10)
  real, pointer, protected_target :: p, parr(:)
  integer, pointer, protected_target :: ip
  character(10), pointer, protected_target :: cp
  complex, pointer, protected_target :: zp
  type(t), pointer, protected_target :: tp
  type(t) :: tv
  real, pointer :: w
  real, pointer, protected_target :: lp
  type(tpp), pointer, protected_target :: tppp
  class(t), pointer, protected_target :: clp
 contains
  real function f0()
    f0 = 0.
  end
  function f() result(r)
    real, pointer, protected_target :: r
    r => tgt
  end
  subroutine definitions
    integer :: j
    namelist /nml/ p
    p => tgt
    parr => tgtarr
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    p = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'parr' is a PROTECTED_TARGET pointer whose target may not be defined through it
    parr(1) = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'parr' is a PROTECTED_TARGET pointer whose target may not be defined through it
    parr(2:3) = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'zp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    zp%re = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'cp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    cp(1:2) = 'ab'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'tp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    tp%x = 1.
    ! The target of an ordinary pointer component is not part of the
    ! target of 'tp'.
    tp%q = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'c' is a PROTECTED_TARGET pointer whose target may not be defined through it
    tv%c = 1.
    tv%c => tgt
    ! Pointer association does not define the nonpointer value of 'tp'
    ! (J3/26-007r1 20.6.1 paragraph 4).
    tp%q => tgt
    tp%c => tgt
    nullify(p)
    nullify(tp%q)
    tppp%pp => f0
    if (.false.) then
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
      p = 2.
    end if
    !ERROR: Input variable 'p' is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    read(*,*) p
    !ERROR: Internal file variable 'cp' is not definable
    !BECAUSE: 'cp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    write(cp,*) 1
    !ERROR: NAMELIST input group must not contain undefinable item 'p'
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    read(*,nml=nml)
    !ERROR: 'ip' may not be used as a DO variable
    !BECAUSE: 'ip' is a PROTECTED_TARGET pointer whose target may not be defined through it
    do ip = 1, 2
    end do
    !ERROR: STAT variable 'ip' is not definable
    !BECAUSE: 'ip' is a PROTECTED_TARGET pointer whose target may not be defined through it
    allocate(w, stat=ip)
    !ERROR: IOLENGTH variable is not definable
    !BECAUSE: 'ip' is a PROTECTED_TARGET pointer whose target may not be defined through it
    inquire(iolength=ip) tgt
    inquire(iolength=j) tgt
    associate (a => p, b => tp%q, c => parr(1:2))
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
      a = 1.
      b = 1.
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'parr' is a PROTECTED_TARGET pointer whose target may not be defined through it
      c = 1.
    end associate
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'r' is a PROTECTED_TARGET pointer whose target may not be defined through it
    f() = 1.
    associate (a => f())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'r' is a PROTECTED_TARGET pointer whose target may not be defined through it
      a = 1.
    end associate
    select type (y => clp)
    type is (t)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'clp' is a PROTECTED_TARGET pointer whose target may not be defined through it
      y%x = 1.
    end select
    !ERROR: 'lp' may not appear in a locality-spec because it is not definable
    !BECAUSE: 'lp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    do concurrent (j=1:2) local(lp)
    end do
  end
  subroutine allocations
    !ERROR: A PROTECTED_TARGET pointer may be allocated only by an ALLOCATE statement with SOURCE=
    allocate(p)
    !ERROR: A PROTECTED_TARGET pointer may be allocated only by an ALLOCATE statement with SOURCE=
    allocate(p, mold=1.)
    allocate(p, source=1.)
    allocate(parr, source=tgtarr)
    !ERROR: A PROTECTED_TARGET pointer may be allocated only by an ALLOCATE statement with SOURCE=
    allocate(tv%c)
    allocate(tv%c, source=1.)
    allocate(tp%q)
    allocate(tp%c, source=1.)
    !ERROR: A PROTECTED_TARGET pointer may be allocated only by an ALLOCATE statement with SOURCE=
    allocate(tp%c)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: 'tp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    allocate(tp%a)
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    deallocate(p)
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'c' is a PROTECTED_TARGET pointer whose target may not be defined through it
    deallocate(tv%c)
    deallocate(tp%q)
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'c' is a PROTECTED_TARGET pointer whose target may not be defined through it
    deallocate(tp%c)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: 'tp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    deallocate(tp%a)
  end
  subroutine ranks(ap)
    real, pointer, protected_target, intent(in out) :: ap(..)
    select rank (x => ap)
    rank (0)
      x => tgt
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'ap' is a PROTECTED_TARGET pointer whose target may not be defined through it
      x = 1.
    rank (1)
      !ERROR: A PROTECTED_TARGET pointer may be allocated only by an ALLOCATE statement with SOURCE=
      allocate(x(2))
      allocate(x, source=tgtarr)
    end select
  end
  subroutine dummies(dp, dpin)
    real, pointer, protected_target, intent(in out) :: dp
    real, pointer, protected_target, intent(in) :: dpin
    real :: x
    dp => tgt
    x = dpin + 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'dp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    dp = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'dpin' is a PROTECTED_TARGET pointer whose target may not be defined through it
    dpin = 1.
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'dpin' is an INTENT(IN) dummy argument
    dpin => tgt
  end
end

module m2
  use m, only: r => p, tgt
 contains
  subroutine renamed
    r => tgt
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    r = 1.
  end
end

module m3
  use, intrinsic :: iso_fortran_env, only: event_type, lock_type, &
      notify_type, team_type
  type te
    type(event_type), pointer, protected_target :: ev
  end type
  type tl
    type(lock_type), pointer, protected_target :: lk
  end type
  type tn
    type(notify_type), pointer, protected_target :: nv
  end type
  type(team_type), pointer, protected_target :: tm
  type(team_type), pointer :: tmw
  logical, pointer, protected_target :: lp
  logical, pointer :: lpw
  integer, pointer, protected_target :: ip
  integer, pointer :: ipw
  type(lock_type), save :: lck[*]
  real, save :: a[*]
 contains
  subroutine image_control(xe, xl, xn)
    type(te) :: xe
    type(tl) :: xl
    type(tn) :: xn
    real :: x
    !ERROR: Team variable is not definable
    !BECAUSE: 'tm' is a PROTECTED_TARGET pointer whose target may not be defined through it
    form team (1, tm)
    form team (1, tmw)
    !ERROR: ACQUIRED_LOCK= variable is not definable
    !BECAUSE: 'lp' is a PROTECTED_TARGET pointer whose target may not be defined through it
    lock (lck, acquired_lock=lp)
    lock (lck, acquired_lock=lpw)
    !ERROR: Event variable is not definable
    !BECAUSE: 'ev' is a PROTECTED_TARGET pointer whose target may not be defined through it
    event post (xe%ev)
    !ERROR: Event variable is not definable
    !BECAUSE: 'ev' is a PROTECTED_TARGET pointer whose target may not be defined through it
    event wait (xe%ev)
    !ERROR: Lock variable is not definable
    !BECAUSE: 'lk' is a PROTECTED_TARGET pointer whose target may not be defined through it
    lock (xl%lk)
    !ERROR: Lock variable is not definable
    !BECAUSE: 'lk' is a PROTECTED_TARGET pointer whose target may not be defined through it
    unlock (xl%lk)
    !ERROR: Image selector STAT= variable is not definable
    !BECAUSE: 'ip' is a PROTECTED_TARGET pointer whose target may not be defined through it
    x = a[1, stat=ip]
    x = a[1, stat=ipw]
    !ERROR: NOTIFY= variable is not definable
    !BECAUSE: 'nv' is a PROTECTED_TARGET pointer whose target may not be defined through it
    a[1, notify=xn%nv] = 1.
    !ERROR: STAT variable 'ip' is not definable
    !BECAUSE: 'ip' is a PROTECTED_TARGET pointer whose target may not be defined through it
    sync all (stat=ip)
  end
end
