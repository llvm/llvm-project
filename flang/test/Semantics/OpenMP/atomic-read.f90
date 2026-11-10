!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00
  integer :: x, v
  ! The end-directive is optional in ATOMIC READ. Expect no diagnostics.
  !$omp atomic read
  v = x

  !$omp atomic read
  v = x
  !$omp end atomic
end

subroutine f01
  integer, pointer :: x, v
  ! Intrinsic assignment and pointer assignment are both ok. Expect no
  ! diagnostics.
  !$omp atomic read
  v = x

  !$omp atomic read
  v => x
end

subroutine f02(i)
  integer :: i, v
  interface
    function p(i)
      integer, pointer :: p
      integer :: i
    end
  end interface

  ! Atomic variable can be a function reference. Expect no diagostics.
  !$omp atomic read
  v = p(i)
end

subroutine f03
  integer :: x(3), y(5), v(3)

  !$omp atomic read
  !ERROR: Atomic variable x should be a scalar
  !ERROR: Capture variable v should be a scalar
  v = x

  !$omp atomic read
  !ERROR: Atomic variable y(2_8:4_8:1_8) should be a scalar
  !ERROR: Capture variable v should be a scalar
  v = y(2:4)
end

subroutine f04
  integer :: x, y(3), v

  !$omp atomic read
  !ERROR: Within atomic operation x and x access the same storage
  x = x

  ! Accessing same array, but not the same storage. Expect no diagnostics.
  !$omp atomic read
  y(1) = y(2)
end

subroutine f05
  integer :: x, v

  !$omp atomic read
  !ERROR: Atomic expression x+1_4 should be a variable
  v = x + 1
end

subroutine f06
  character :: x, v

  !$omp atomic read
  !ERROR: Atomic variable x cannot have CHARACTER type
  !ERROR: Capture variable v cannot have CHARACTER type
  v = x
end

subroutine f07
  integer, allocatable :: x
  integer :: v

  allocate(x)

  !$omp atomic read
  !ERROR: Atomic variable x cannot be ALLOCATABLE
  v = x
end

subroutine f08
  type :: struct
    integer :: m
  end type
  type(struct) :: x, v

  !$omp atomic read
  !ERROR: Atomic variable x should have an intrinsic type
  !ERROR: Capture variable v should have an intrinsic type
  v = x
end

subroutine f09(x, v)
  class(*), pointer :: x, v

  !$omp atomic read
  !ERROR: Atomic variable x cannot be a pointer to a polymorphic type
  !ERROR: Capture variable v cannot be a pointer to a polymorphic type
  v => x
end

subroutine f10(x, v)
  type struct(length)
    integer, len :: length
  end type
  type(struct(*)), pointer :: x, v

  !$omp atomic read
  !ERROR: Atomic variable x is a pointer to a type with non-constant length parameter
  !ERROR: Capture variable v is a pointer to a type with non-constant length parameter
  v => x
end

subroutine f11
  integer :: x, v(3)
  integer, allocatable :: w

  !$omp atomic read
  !ERROR: Capture variable v should be a scalar
  v = x

  !$omp atomic read
  !ERROR: Capture variable v(1_8:2_8:1_8) should be a scalar
  v(1:2) = x

  ! An array element is a scalar, and v can be ALLOCATABLE. Expect no
  ! diagnostics.
  !$omp atomic read
  v(1) = x

  !$omp atomic read
  w = x
end

subroutine f12(x, v)
  integer, pointer :: x
  class(*), pointer :: v

  !$omp atomic read
  !ERROR: Capture variable v cannot be a pointer to a polymorphic type
  v => x
end

module m13
  type t
    integer :: i
  end type
  interface assignment(=)
    module procedure assign_t
  end interface
contains
  subroutine assign_t(a, b)
    type(t), intent(out) :: a
    integer, intent(in) :: b
    a%i = b
  end
end

subroutine f13(x)
  use m13
  integer :: x
  type(t) :: v

  !$omp atomic read
  !ERROR: Capture variable v should have an intrinsic type
  v = x
end

subroutine f14(x, c)
  integer :: x, v
  character :: c
  interface
    function q(c) result(r)
      character :: c
      integer, pointer :: r
    end
  end interface

  ! x and v can be function references. The types of the arguments don't
  ! matter. Expect no diagnostics.
  !$omp atomic read
  v = q(c)

  !$omp atomic read
  q(c) = x
end

subroutine f15(x, v)
  type t
    integer :: i
  end type
  type(t), pointer :: x, v
  integer, allocatable :: y
  integer :: w

  ! Variables in a parallel region are checked the same way as outside it.
  !$omp parallel
  !$omp atomic read
  v => x

  !$omp atomic read
  !ERROR: Atomic variable y cannot be ALLOCATABLE
  w = y
  !$omp end parallel
end

subroutine f16(x, y)
  class(*), pointer :: x
  integer, allocatable :: y
  integer :: v

  ! An associate name doesn't have the POINTER or ALLOCATABLE attribute of
  ! its selector. Expect no diagnostics.
  select type (a => x)
  type is (integer)
    !$omp atomic read
    v = a
  end select

  associate (b => y)
    !$omp atomic read
    v = b
  end associate
end
