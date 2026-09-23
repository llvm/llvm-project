! RUN: %python %S/test_errors.py %s %flang_fc1
! Valid-code guard for IsProcedure(Expr): function references with every
! callee/result flavor that the predicate's function-reference branch walks
! (statement functions, data-pointer results, procedure pointers, dummy
! procedures, type-bound and generic callees, ENTRY results, recursive
! references, and procedure-pointer-valued function results in their
! conforming uses) must all still be accepted.
! No errors expected anywhere in this file.

module m
  integer, target :: itgt = 9
  type t
  contains
    procedure :: tbp
  end type
  interface gen
    module procedure fi, fr
  end interface
  interface operator(.plus.)
    module procedure add
  end interface
contains
  integer function fi(i)
    integer, intent(in) :: i
    fi = i
  end function
  real function fr(r)
    real, intent(in) :: r
    fr = r
  end function
  integer function tbp(this)
    class(t), intent(in) :: this
    tbp = 3
  end function
  integer function add(a, b)
    integer, intent(in) :: a, b
    add = a + b
  end function
  function fp() result(r)
    integer, pointer :: r
    allocate(r)
    r = 5
  end function
  function fpa() result(r)
    integer, pointer :: r(:)
    allocate(r(3))
    r = 7
  end function
  character(5) function cf()
    cf = 'hello'
  end function
  function af() result(r)
    integer :: r(3)
    r = [1, 2, 3]
  end function
  elemental integer function ef(i)
    integer, intent(in) :: i
    ef = i * 2
  end function
  integer function fent()
    integer :: gent
    fent = 1
    return
  entry gent()
    gent = 2
  end function
  function polya() result(r)
    class(*), allocatable :: r
    r = 42
  end function
  function polyp() result(r)
    class(*), pointer :: r
    r => itgt
  end function
  integer function target1()
    target1 = 7
  end function
  function getp() result(r)
    procedure(target1), pointer :: r
    r => target1
  end function
end module

recursive function fact(n) result(r)
  integer, intent(in) :: n
  integer :: r
  if (n <= 1) then
    r = 1
  else
    r = n * fact(n - 1)
  end if
  print *, fact(1)
end function

subroutine dummyproc(g)
  use m, only: fi
  procedure(fi) :: g
  print *, g(1)
end

program p
  use m
  use m, only: renamed => fi
  procedure(target1), pointer :: q
  type(t) :: x
  real :: sf, xx
  external impf
  real impf
  sf(xx) = xx + 1.0
  print *, sf(2.0)
  print *, fp(), size(fpa())
  print *, x%tbp()
  print *, gen(1), gen(1.5)
  print *, 1 .plus. 2
  print *, renamed(4)
  print *, fent(), gent()
  print *, cf(), af(), ef([1, 2, 3]), size(af())
  print *, impf()
  select type (y => polya())
  type is (integer)
    print *, y
  end select
  select type (y => polyp())
  type is (integer)
    print *, y
  end select
  q => getp()
  print *, q()
  print *, associated(getp(), target1)
  call take(getp())
contains
  subroutine take(d)
    procedure(target1) :: d
    print *, d()
  end subroutine
end
