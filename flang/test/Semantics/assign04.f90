! RUN: %python %S/test_errors.py %s %flang_fc1
! 9.4.5
subroutine s1
  type :: t(k, l)
    integer, kind :: k
    integer, len :: l
  end type
  type(t(1, 2)) :: x
  !ERROR: Assignment to constant 'x%k' is not allowed
  x%k = 4
  !ERROR: Assignment to constant 'x%l' is not allowed
  x%l = 3
end

! C901
subroutine s2(x)
  !ERROR: A dummy argument may not also be a named constant
  real, parameter :: x = 0.0
  real, parameter :: a(*) = [1, 2, 3]
  character, parameter :: c(2) = "ab"
  integer :: i
  !ERROR: Assignment to constant 'x' is not allowed
  x = 2.0
  i = 2
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'a' is not a variable
  a(i) = 3.0
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'a' is not a variable
  a(i:i+1) = [4, 5]
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'c' is not a variable
  c(i:2) = "cd"
end

! C901
subroutine s3
  type :: t
    integer :: a(2)
    integer :: b
  end type
  type(t) :: x
  type(t), parameter :: y = t([1,2], 3)
  integer :: i = 1
  x%a(i) = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'y' is not a variable
  y%a(i) = 2
  x%b = 4
  !ERROR: Assignment to constant 'y%b' is not allowed
  y%b = 5
end

! C844
subroutine s4
  type :: t
    integer :: a(2)
  end type
contains
  subroutine s(x, c)
    type(t), intent(in) :: x
    character(10), intent(in) :: c
    type(t) :: y
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'x' is an INTENT(IN) dummy argument
    x = y
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'x' is an INTENT(IN) dummy argument
    x%a(1) = 2
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'c' is an INTENT(IN) dummy argument
    c(2:3) = "ab"
  end
end

! 8.5.15(2)
module m5
  real :: x
  real, protected :: y
  real, private :: z
  type :: t
    real :: a
  end type
  type(t), protected :: b
end
subroutine s5()
  use m5
  implicit none
  x = 1.0
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'y' is protected in this scope
  y = 2.0
  !ERROR: No explicit type declared for 'z'
  z = 3.0
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'b' is protected in this scope
  b%a = 1.0
end

subroutine s6(x)
  integer :: x(*)
  x(1:3) = [1, 2, 3]
  x(:3) = [1, 2, 3]
  !ERROR: Assumed-size array 'x' must have explicit final subscript upper bound value
  x(:) = [1, 2, 3]
  !ERROR: Whole assumed-size array 'x' may not appear here without subscripts
  x = [1, 2, 3]
  associate (y => x) ! ok
    !ERROR: Whole assumed-size array 'y' may not appear here without subscripts
    y = [1, 2, 3]
  end associate
  !ERROR: Whole assumed-size array 'x' may not appear here without subscripts
  associate (y => (x))
  end associate
end

module m7
  type :: t
    integer :: i
  end type
contains
  subroutine s7(x)
    type(t) :: x(*)
    x(:3)%i = [1, 2, 3]
    !ERROR: Whole assumed-size array 'x' may not appear here without subscripts
    x%i = [1, 2, 3]
  end
end

subroutine s7
  integer :: a(10), v(10)
  a(v(:)) = 1  ! vector subscript is ok
end

subroutine s8
  !ERROR: Assignment to procedure 's8' is not allowed
  s8 = 1.0
end

real function f9() result(r)
  !ERROR: Assignment to procedure 'f9' is not allowed
  f9 = 1.0
end

subroutine s9
  real f9a
  !ERROR: Assignment to procedure 'f9a' is not allowed
  f9a = 1.0
  print *, f9a(1)
end

!ERROR: No explicit type declared for dummy argument 'n'
subroutine s10(a, n)
  implicit none
  real a(n)
  a(1:n) = 0.0  ! should not get a second error here
end

subroutine s11
  intrinsic :: sin
  real :: a
  !ERROR: Function call must have argument list
  a = sin
  !ERROR: Subroutine name is not allowed here
  a = s11
end

subroutine s12()
  type dType(l1, k1, l2, k2)
    integer, len :: l1
    integer, kind :: k1
    integer, len :: l2
    integer, kind :: k2
  end type

  contains
    subroutine sub(arg1, arg2, arg3)
      integer :: arg1
      type(dType(arg1, 2, *, 4)) :: arg2
      type(dType(*, 2, arg1, 4)) :: arg3
      type(dType(1, 2, 3, 4)) :: local1
      type(dType(1, 2, 3, 4)) :: local2
      type(dType(1, 2, arg1, 4)) :: local3
      type(dType(9, 2, 3, 4)) :: local4
      type(dType(1, 9, 3, 4)) :: local5

      arg2 = arg3
      arg2 = local1
      arg3 = local1
      local1 = local2
      local2 = local3
      !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(dtype(k1=2_4,k2=4_4,l1=1_4,l2=3_4)) and TYPE(dtype(k1=2_4,k2=4_4,l1=9_4,l2=3_4))
      local1 = local4 ! mismatched constant KIND type parameter
      !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(dtype(k1=2_4,k2=4_4,l1=1_4,l2=3_4)) and TYPE(dtype(k1=9_4,k2=4_4,l1=1_4,l2=3_4))
      local1 = local5 ! mismatched constant LEN type parameter
    end subroutine sub
end subroutine s12

subroutine s13()
  interface assignment(=)
    procedure :: cToR, cToRa, cToI
  end interface
  real :: x(1)
  integer :: n(1)
  x='0' ! fine
  n='0' ! fine
  !ERROR: Defined assignment in WHERE must be elemental, but 'ctora' is not
  where ([1==1]) x='*'
  where ([1==1]) n='*' ! fine
  forall (j=1:1)
    !ERROR: The mask or variable must not be scalar
    where (j==1)
      !ERROR: Defined assignment in WHERE must be elemental, but 'ctor' is not
      !ERROR: The mask or variable must not be scalar
      x(j)='?'
      !ERROR: The mask or variable must not be scalar
      n(j)='?'
    !ERROR: The mask or variable must not be scalar
    elsewhere (.false.)
      !ERROR: Defined assignment in WHERE must be elemental, but 'ctor' is not
      !ERROR: The mask or variable must not be scalar
      x(j)='1'
      !ERROR: The mask or variable must not be scalar
      n(j)='1'
    elsewhere
      !ERROR: Defined assignment in WHERE must be elemental, but 'ctor' is not
      !ERROR: The mask or variable must not be scalar
      x(j)='9'
      !ERROR: The mask or variable must not be scalar
      n(j)='9'
    end where
  end forall
  x='0' ! still fine
  n='0' ! still fine
 contains
  subroutine cToR(x, c)
    real, intent(out) :: x
    character, intent(in) :: c
  end subroutine
  subroutine cToRa(x, c)
    real, intent(out) :: x(:)
    character, intent(in) :: c
  end subroutine
  elemental subroutine cToI(n, c)
    integer, intent(out) :: n
    character, intent(in) :: c
  end subroutine
end subroutine s13

module m14
  type t1
    integer, pointer :: p
   contains
    procedure definedAsst1
    generic :: assignment(=) => definedAsst1
  end type
  type t2
    integer, pointer :: p
  end type
  interface assignment(=)
    module procedure definedAsst2
  end interface
  type t3
    integer, pointer :: p
  end type
 contains
  pure subroutine definedAsst1(lhs,rhs)
    class(t1), intent(in out) :: lhs
    class(t1), intent(in) :: rhs
  end subroutine
  pure subroutine definedAsst2(lhs,rhs)
    type(t2), intent(out) :: lhs
    type(t2), intent(in) :: rhs
  end subroutine
  pure subroutine test(y1,y2,y3)
    type(t1) x1
    type(t1), intent(in) :: y1
    type(t2) x2
    type(t2), intent(in) :: y2
    type(t3) x3
    type(t3), intent(in) :: y3
    x1 = y1 ! fine due to not being intrinsic assignment
    x2 = y2 ! fine due to not being intrinsic assignment
    !ERROR: A pure subprogram may not copy the value of 'y3' because it is an INTENT(IN) dummy argument and has the POINTER potential subobject component '%p'
    x3 = y3
  end subroutine
end module m14
