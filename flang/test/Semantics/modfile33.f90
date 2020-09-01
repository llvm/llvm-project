! RUN: %S/test_modfile.sh %s %t %f18 -flogical-abbreviations -fxor-operator

! Resolution of user-defined operators in expressions.
! Test by using generic function in a specification expression that needs
! to be written to a .mod file.

! Numeric operators
module m1
  type :: t
    sequence
    logical :: x
  end type
  interface operator(+)
    pure integer(8) function add_ll(x, y)
      logical, intent(in) :: x, y
    end
    pure integer(8) function add_li(x, y)
      logical, intent(in) :: x
      integer, intent(in) :: y
    end
    pure integer(8) function add_tt(x, y)
      import :: t
      type(t), intent(in) :: x, y
    end
  end interface
  interface operator(/)
    pure integer(8) function div_tz(x, y)
      import :: t
      type(t), intent(in) :: x
      complex, intent(in) :: y
    end
    pure integer(8) function div_ct(x, y)
      import :: t
      character(10), intent(in) :: x
      type(t), intent(in) :: y
    end
  end interface
contains
  subroutine s1(x, y, z)
    logical :: x, y
    real :: z(x + y)  ! resolves to add_ll
  end
  subroutine s2(x, y, z)
    logical :: x
    integer :: y
    real :: z(x + y)  ! resolves to add_li
  end
  subroutine s3(x, y, z)
    type(t) :: x
    complex :: y
    real :: z(x / y)  ! resolves to div_tz
  end
  subroutine s4(x, y, z)
    character(10) :: x
    type(t) :: y
    real :: z(x / y)  ! resolves to div_ct
  end
end

!Expect: m1.mod
!module m1
! type :: t
!  sequence
!  logical(4) :: x
! end type
! interface operator(+)
!  procedure :: add_ll
!  procedure :: add_li
!  procedure :: add_tt
! end interface
! interface
!  pure function add_ll(x, y)
!   logical(4), intent(in) :: x
!   logical(4), intent(in) :: y
!   integer(8) :: add_ll
!  end
! end interface
! interface
!  pure function add_li(x, y)
!   logical(4), intent(in) :: x
!   integer(4), intent(in) :: y
!   integer(8) :: add_li
!  end
! end interface
! interface
!  pure function add_tt(x, y)
!   import :: t
!   type(t), intent(in) :: x
!   type(t), intent(in) :: y
!   integer(8) :: add_tt
!  end
! end interface
! interface operator(/)
!  procedure :: div_tz
!  procedure :: div_ct
! end interface
! interface
!  pure function div_tz(x, y)
!   import :: t
!   type(t), intent(in) :: x
!   complex(4), intent(in) :: y
!   integer(8) :: div_tz
!  end
! end interface
! interface
!  pure function div_ct(x, y)
!   import :: t
!   character(10_4, 1), intent(in) :: x
!   type(t), intent(in) :: y
!   integer(8) :: div_ct
!  end
! end interface
!contains
! subroutine s1(x, y, z)
!  logical(4) :: x
!  logical(4) :: y
!  real(4) :: z(1_8:add_ll(x, y))
! end
! subroutine s2(x, y, z)
!  logical(4) :: x
!  integer(4) :: y
!  real(4) :: z(1_8:add_li(x, y))
! end
! subroutine s3(x, y, z)
!  type(t) :: x
!  complex(4) :: y
!  real(4) :: z(1_8:div_tz(x, y))
! end
! subroutine s4(x, y, z)
!  character(10_4, 1) :: x
!  type(t) :: y
!  real(4) :: z(1_8:div_ct(x, y))
! end
!end

! Logical operators
module m2
  type :: t
    sequence
    logical :: x
  end type
  interface operator(.And.)
    pure integer(8) function and_ti(x, y)
      import :: t
      type(t), intent(in) :: x
      integer, intent(in) :: y
    end
    pure integer(8) function and_li(x, y)
      logical, intent(in) :: x
      integer, intent(in) :: y
    end
  end interface
  ! Alternative spelling of .AND.
  interface operator(.a.)
    pure integer(8) function and_tt(x, y)
      import :: t
      type(t), intent(in) :: x, y
    end
  end interface
  interface operator(.x.)
    pure integer(8) function neqv_tt(x, y)
      import :: t
      type(t), intent(in) :: x, y
    end
  end interface
  interface operator(.neqv.)
    pure integer(8) function neqv_rr(x, y)
      real, intent(in) :: x, y
    end
  end interface
contains
  subroutine s1(x, y, z)
    type(t) :: x
    integer :: y
    real :: z(x .and. y)  ! resolves to and_ti
  end
  subroutine s2(x, y, z)
    logical :: x
    integer :: y
    real :: z(x .a. y)  ! resolves to and_li
  end
  subroutine s3(x, y, z)
    type(t) :: x, y
    real :: z(x .and. y)  ! resolves to and_tt
  end
  subroutine s4(x, y, z)
    type(t) :: x, y
    real :: z(x .neqv. y)  ! resolves to neqv_tt
  end
  subroutine s5(x, y, z)
    real :: x, y
    real :: z(x .xor. y)  ! resolves to neqv_rr
  end
end

!Expect: m2.mod
!module m2
! type :: t
!  sequence
!  logical(4) :: x
! end type
! interface operator( .and.)
!  procedure :: and_ti
!  procedure :: and_li
!  procedure :: and_tt
! end interface
! interface
!  pure function and_ti(x, y)
!   import :: t
!   type(t), intent(in) :: x
!   integer(4), intent(in) :: y
!   integer(8) :: and_ti
!  end
! end interface
! interface
!  pure function and_li(x, y)
!   logical(4), intent(in) :: x
!   integer(4), intent(in) :: y
!   integer(8) :: and_li
!  end
! end interface
! interface
!  pure function and_tt(x, y)
!   import :: t
!   type(t), intent(in) :: x
!   type(t), intent(in) :: y
!   integer(8) :: and_tt
!  end
! end interface
! interface operator(.x.)
!  procedure :: neqv_tt
!  procedure :: neqv_rr
! end interface
! interface
!  pure function neqv_tt(x, y)
!   import :: t
!   type(t), intent(in) :: x
!   type(t), intent(in) :: y
!   integer(8) :: neqv_tt
!  end
! end interface
! interface
!  pure function neqv_rr(x, y)
!   real(4), intent(in) :: x
!   real(4), intent(in) :: y
!   integer(8) :: neqv_rr
!  end
! end interface
!contains
! subroutine s1(x, y, z)
!  type(t) :: x
!  integer(4) :: y
!  real(4) :: z(1_8:and_ti(x, y))
! end
! subroutine s2(x, y, z)
!  logical(4) :: x
!  integer(4) :: y
!  real(4) :: z(1_8:and_li(x, y))
! end
! subroutine s3(x, y, z)
!  type(t) :: x
!  type(t) :: y
!  real(4) :: z(1_8:and_tt(x, y))
! end
! subroutine s4(x, y, z)
!  type(t) :: x
!  type(t) :: y
!  real(4) :: z(1_8:neqv_tt(x, y))
! end
! subroutine s5(x, y, z)
!  real(4) :: x
!  real(4) :: y
!  real(4) :: z(1_8:neqv_rr(x, y))
! end
!end

! Relational operators
module m3
  type :: t
    sequence
    logical :: x
  end type
  interface operator(<>)
    pure integer(8) function ne_it(x, y)
      import :: t
      integer, intent(in) :: x
      type(t), intent(in) :: y
    end
  end interface
  interface operator(/=)
    pure integer(8) function ne_tt(x, y)
      import :: t
      type(t), intent(in) :: x, y
    end
  end interface
  interface operator(.ne.)
    pure integer(8) function ne_ci(x, y)
      character(len=*), intent(in) :: x
      integer, intent(in) :: y
    end
  end interface
contains
  subroutine s1(x, y, z)
    integer :: x
    type(t) :: y
    real :: z(x /= y)  ! resolves to ne_it
  end
  subroutine s2(x, y, z)
    type(t) :: x
    type(t) :: y
    real :: z(x .ne. y)  ! resolves to ne_tt
  end
  subroutine s3(x, y, z)
    character(len=*) :: x
    integer :: y
    real :: z(x <> y)  ! resolves to ne_ci
  end
end

!Expect: m3.mod
!module m3
! type :: t
!  sequence
!  logical(4) :: x
! end type
! interface operator(<>)
!  procedure :: ne_it
!  procedure :: ne_tt
!  procedure :: ne_ci
! end interface
! interface
!  pure function ne_it(x, y)
!   import :: t
!   integer(4), intent(in) :: x
!   type(t), intent(in) :: y
!   integer(8) :: ne_it
!  end
! end interface
! interface
!  pure function ne_tt(x, y)
!   import :: t
!   type(t), intent(in) :: x
!   type(t), intent(in) :: y
!   integer(8) :: ne_tt
!  end
! end interface
! interface
!  pure function ne_ci(x, y)
!   character(*, 1), intent(in) :: x
!   integer(4), intent(in) :: y
!   integer(8) :: ne_ci
!  end
! end interface
!contains
! subroutine s1(x, y, z)
!  integer(4) :: x
!  type(t) :: y
!  real(4) :: z(1_8:ne_it(x, y))
! end
! subroutine s2(x, y, z)
!  type(t) :: x
!  type(t) :: y
!  real(4) :: z(1_8:ne_tt(x, y))
! end
! subroutine s3(x, y, z)
!  character(*, 1) :: x
!  integer(4) :: y
!  real(4) :: z(1_8:ne_ci(x, y))
! end
!end

! Concatenation
module m4
  type :: t
    sequence
    logical :: x
  end type
  interface operator(//)
    pure integer(8) function concat_12(x, y)
      character(len=*,kind=1), intent(in) :: x
      character(len=*,kind=2), intent(in) :: y
    end
    pure integer(8) function concat_int_real(x, y)
      integer, intent(in) :: x
      real, intent(in) :: y
    end
  end interface
contains
  subroutine s1(x, y, z)
    character(len=*,kind=1) :: x
    character(len=*,kind=2) :: y
    real :: z(x // y)  ! resolves to concat_12
  end
  subroutine s2(x, y, z)
    integer :: x
    real :: y
    real :: z(x // y)  ! resolves to concat_int_real
  end
end
!Expect: m4.mod
!module m4
! type :: t
!  sequence
!  logical(4) :: x
! end type
! interface operator(//)
!  procedure :: concat_12
!  procedure :: concat_int_real
! end interface
! interface
!  pure function concat_12(x, y)
!   character(*, 1), intent(in) :: x
!   character(*, 2), intent(in) :: y
!   integer(8) :: concat_12
!  end
! end interface
! interface
!  pure function concat_int_real(x, y)
!   integer(4), intent(in) :: x
!   real(4), intent(in) :: y
!   integer(8) :: concat_int_real
!  end
! end interface
!contains
! subroutine s1(x, y, z)
!  character(*, 1) :: x
!  character(*, 2) :: y
!  real(4) :: z(1_8:concat_12(x, y))
! end
! subroutine s2(x, y, z)
!  integer(4) :: x
!  real(4) :: y
!  real(4) :: z(1_8:concat_int_real(x, y))
! end
!end

! Unary operators
module m5
  type :: t
  end type
  interface operator(+)
    pure integer(8) function plus_l(x)
      logical, intent(in) :: x
    end
  end interface
  interface operator(-)
    pure integer(8) function minus_t(x)
      import :: t
      type(t), intent(in) :: x
    end
  end interface
  interface operator(.not.)
    pure integer(8) function not_t(x)
      import :: t
      type(t), intent(in) :: x
    end
    pure integer(8) function not_real(x)
      real, intent(in) :: x
    end
  end interface
contains
  subroutine s1(x, y)
    logical :: x
    real :: y(+x)  ! resolves_to plus_l
  end
  subroutine s2(x, y)
    type(t) :: x
    real :: y(-x)  ! resolves_to minus_t
  end
  subroutine s3(x, y)
    type(t) :: x
    real :: y(.not. x)  ! resolves to not_t
  end
  subroutine s4(x, y)
    real :: y(.not. x)  ! resolves to not_real
  end
end

!Expect: m5.mod
!module m5
! type :: t
! end type
! interface operator(+)
!  procedure :: plus_l
! end interface
! interface
!  pure function plus_l(x)
!   logical(4), intent(in) :: x
!   integer(8) :: plus_l
!  end
! end interface
! interface operator(-)
!  procedure :: minus_t
! end interface
! interface
!  pure function minus_t(x)
!   import :: t
!   type(t), intent(in) :: x
!   integer(8) :: minus_t
!  end
! end interface
! interface operator( .not.)
!  procedure :: not_t
!  procedure :: not_real
! end interface
! interface
!  pure function not_t(x)
!   import :: t
!   type(t), intent(in) :: x
!   integer(8) :: not_t
!  end
! end interface
! interface
!  pure function not_real(x)
!   real(4), intent(in) :: x
!   integer(8) :: not_real
!  end
! end interface
!contains
! subroutine s1(x, y)
!  logical(4) :: x
!  real(4) :: y(1_8:plus_l(x))
! end
! subroutine s2(x, y)
!  type(t) :: x
!  real(4) :: y(1_8:minus_t(x))
! end
! subroutine s3(x, y)
!  type(t) :: x
!  real(4) :: y(1_8:not_t(x))
! end
! subroutine s4(x, y)
!  real(4) :: x
!  real(4) :: y(1_8:not_real(x))
! end
!end

! Resolved based on shape
module m6
  interface operator(+)
    pure integer(8) function add(x, y)
      real, intent(in) :: x(:, :)
      real, intent(in) :: y(:, :, :)
    end
  end interface
contains
  subroutine s1(n, x, y, z, a, b)
    integer(8) :: n
    real :: x
    real :: y(4, n)
    real :: z(2, 2, 2)
    real :: a(size(x+y))  ! intrinsic +
    real :: b(y+z)  ! resolves to add
  end
end

!Expect: m6.mod
!module m6
! interface operator(+)
!  procedure :: add
! end interface
! interface
!  pure function add(x, y)
!   real(4), intent(in) :: x(:, :)
!   real(4), intent(in) :: y(:, :, :)
!   integer(8) :: add
!  end
! end interface
!contains
! subroutine s1(n, x, y, z, a, b)
!  integer(8) :: n
!  real(4) :: x
!  real(4) :: y(1_8:4_8, 1_8:n)
!  real(4) :: z(1_8:2_8, 1_8:2_8, 1_8:2_8)
!  real(4) :: a(1_8:int(int(4_8*(n-1_8+1_8),kind=4),kind=8))
!  real(4) :: b(1_8:add(y, z))
! end
!end

! Parameterized derived type
module m7
  type :: t(k)
    integer, kind :: k
    real(k) :: a
  end type
  interface operator(+)
    pure integer(8) function f1(x, y)
      import :: t
      type(t(4)), intent(in) :: x, y
    end
    pure integer(8) function f2(x, y)
      import :: t
      type(t(8)), intent(in) :: x, y
    end
  end interface
contains
  subroutine s1(x, y, z)
    type(t(4)) :: x, y
    real :: z(x + y)  ! resolves to f1
  end
  subroutine s2(x, y, z)
    type(t(8)) :: x, y
    real :: z(x + y)  ! resolves to f2
  end
end

!Expect: m7.mod
!module m7
! type :: t(k)
!  integer(4), kind :: k
!  real(int(int(k,kind=4),kind=8))::a
! end type
! interface operator(+)
!  procedure :: f1
!  procedure :: f2
! end interface
! interface
!  pure function f1(x, y)
!   import :: t
!   type(t(k=4_4)), intent(in) :: x
!   type(t(k=4_4)), intent(in) :: y
!   integer(8) :: f1
!  end
! end interface
! interface
!  pure function f2(x, y)
!   import :: t
!   type(t(k=8_4)), intent(in) :: x
!   type(t(k=8_4)), intent(in) :: y
!   integer(8) :: f2
!  end
! end interface
!contains
! subroutine s1(x, y, z)
!  type(t(k=4_4)) :: x
!  type(t(k=4_4)) :: y
!  real(4) :: z(1_8:f1(x, y))
! end
! subroutine s2(x, y, z)
!  type(t(k=8_4)) :: x
!  type(t(k=8_4)) :: y
!  real(4) :: z(1_8:f2(x, y))
! end
!end
