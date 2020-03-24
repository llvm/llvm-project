! RUN: %S/test_errors.sh %s %flang %t
! 9.4.5
subroutine s1
  type :: t(k, l)
    integer, kind :: k
    integer, len :: l
  end type
  type(t(1, 2)) :: x
  !ERROR: Assignment to constant 'x%k' is not allowed
  x%k = 4
  !ERROR: Left-hand side of assignment is not modifiable
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
  !ERROR: Left-hand side of assignment is not modifiable
  a(i) = 3.0
  !ERROR: Left-hand side of assignment is not modifiable
  a(i:i+1) = [4, 5]
  !ERROR: Left-hand side of assignment is not modifiable
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
  !ERROR: Left-hand side of assignment is not modifiable
  y%a(i) = 2
  x%b = 4
  !ERROR: Left-hand side of assignment is not modifiable
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
    !ERROR: Left-hand side of assignment is not modifiable
    x = y
    !ERROR: Left-hand side of assignment is not modifiable
    x%a(1) = 2
    !ERROR: Left-hand side of assignment is not modifiable
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
  !ERROR: Left-hand side of assignment is not modifiable
  y = 2.0
  !ERROR: No explicit type declared for 'z'
  z = 3.0
  !ERROR: Left-hand side of assignment is not modifiable
  b%a = 1.0
end

subroutine s6(x)
  integer :: x(*)
  x(1:3) = [1, 2, 3]
  x(:3) = [1, 2, 3]
  !ERROR: Assumed-size array 'x' must have explicit final subscript upper bound value
  x(:) = [1, 2, 3]
  !ERROR: Left-hand side of assignment may not be a whole assumed-size array
  x = [1, 2, 3]
end

module m7
  type :: t
    integer :: i
  end type
contains
  subroutine s7(x)
    type(t) :: x(*)
    x(:3)%i = [1, 2, 3]
    !ERROR: Left-hand side of assignment may not be a whole assumed-size array
    x%i = [1, 2, 3]
  end
end

subroutine s7
  integer :: a(10), v(10)
  a(v(:)) = 1  ! vector subscript is ok
end
