! RUN: %S/test_errors.sh %s %t %f18
subroutine s1
  integer i, j
  real r(2)
  !ERROR: Equivalence set must have more than one object
  equivalence(i, j),(r(1))
end

subroutine s2
  integer i
  type t
    integer :: a
    integer :: b(10)
  end type
  type(t) :: x
  !ERROR: Derived type component 'x%a' is not allowed in an equivalence set
  equivalence(x%a, i)
  !ERROR: Derived type component 'x%b(2)' is not allowed in an equivalence set
  equivalence(i, x%b(2))
end

integer function f3(x)
  real x
  !ERROR: Dummy argument 'x' is not allowed in an equivalence set
  equivalence(i, x)
  !ERROR: Function result 'f3' is not allow in an equivalence set
  equivalence(f3, i)
end

subroutine s4
  integer :: y
  !ERROR: Pointer 'x' is not allowed in an equivalence set
  !ERROR: Allocatable variable 'y' is not allowed in an equivalence set
  equivalence(x, y)
  real, pointer :: x
  allocatable :: y
end

subroutine s5
  integer, parameter :: k = 123
  real :: x(10)
  real, save :: y[1:*]
  !ERROR: Coarray 'y' is not allowed in an equivalence set
  equivalence(x, y)
  !ERROR: Variable 'z' with BIND attribute is not allowed in an equivalence set
  equivalence(x, z)
  !ERROR: Variable 'z' with BIND attribute is not allowed in an equivalence set
  equivalence(x(2), z(3))
  real, bind(C) :: z(10)
  !ERROR: Named constant 'k' is not allowed in an equivalence set
  equivalence(x(2), k)
  !ERROR: Variable 'w' in common block with BIND attribute is not allowed in an equivalence set
  equivalence(x(10), w)
  logical :: w(10)
  bind(C, name="c") /c/
  common /c/ w
  integer, target :: u
  !ERROR: Variable 'u' with TARGET attribute is not allowed in an equivalence set
  equivalence(x(1), u)
end

subroutine s6
  type t1
    sequence
    real, pointer :: p
  end type
  type :: t2
    sequence
    type(t1) :: b
  end type
  real :: x0
  type(t1) :: x1
  type(t2) :: x2
  !ERROR: Derived type object 'x1' with pointer ultimate component is not allowed in an equivalence set
  equivalence(x0, x1)
  !ERROR: Derived type object 'x2' with pointer ultimate component is not allowed in an equivalence set
  equivalence(x0, x2)
end

subroutine s7
  type t1
  end type
  real :: x0
  type(t1) :: x1
  !ERROR: Nonsequence derived type object 'x1' is not allowed in an equivalence set
  equivalence(x0, x1)
end

module m8
  real :: x
  real :: y(10)
end
subroutine s8
  use m8
  !ERROR: Use-associated variable 'x' is not allowed in an equivalence set
  equivalence(x, z)
  !ERROR: Use-associated variable 'y' is not allowed in an equivalence set
  equivalence(y(1), z)
end

subroutine s9
  character(10) :: c
  real :: d(10)
  integer, parameter :: n = 2
  integer :: i, j
  !ERROR: Substring with nonconstant bound 'n+j' is not allowed in an equivalence set
  equivalence(c(n+1:n+j), i)
  !ERROR: Substring with zero length is not allowed in an equivalence set
  equivalence(c(n:1), i)
  !ERROR: Array with nonconstant subscript 'j-1' is not allowed in an equivalence set
  equivalence(d(j-1), i)
  !ERROR: Array section 'd(1:n)' is not allowed in an equivalence set
  equivalence(d(1:n), i)
  character(4) :: a(10)
  equivalence(c, a(10)(1:2))
  !ERROR: 'a(10_8)(2_8:2_8)' and 'a(10_8)(1_8:1_8)' cannot have the same first storage unit
  equivalence(c, a(10)(2:3))
end

subroutine s10
  integer, parameter :: i(4) = [1, 2, 3, 4]
  real :: x(10)
  real :: y(4)
  !ERROR: Array with vector subscript 'i' is not allowed in an equivalence set
  equivalence(x(i), y)
end

subroutine s11(n)
  integer :: n
  real :: x(n), y
  !ERROR: Automatic object 'x' is not allowed in an equivalence set
  equivalence(x(1), y)
end

module s12
  real, protected :: a
  integer :: b
  !ERROR: Equivalence set cannot contain 'a' with PROTECTED attribute and 'b' without
  equivalence(a, b)
  !ERROR: Equivalence set cannot contain 'a' with PROTECTED attribute and 'b' without
  equivalence(b, a)
end

module s13
  logical(8) :: a
  character(4) :: b
  type :: t1
    sequence
    complex :: z
  end type
  type :: t2
    sequence
    type(t1) :: w
  end type
  type(t2) :: c
  !ERROR: Equivalence set cannot contain 'b' that is character sequence type and 'a' that is not
  equivalence(a, b)
  !ERROR: Equivalence set cannot contain 'c' that is numeric sequence type and 'a' that is not
  equivalence(c, a)
  double precision :: d
  double complex :: e
  !OK: d and e are considered to be a default kind numeric type
  equivalence(c, d, e)
end

module s14
  real :: a(10), b, c, d
  !ERROR: 'a(2_8)' and 'a(1_8)' cannot have the same first storage unit
  equivalence(a(1), a(2))
  equivalence(b, a(3))
  !ERROR: 'a(4_8)' and 'a(3_8)' cannot have the same first storage unit
  equivalence(a(4), b)
  equivalence(c, a(5))
  !ERROR: 'a(6_8)' and 'a(5_8)' cannot have the same first storage unit
  equivalence(a(6), d)
  equivalence(c, d)
end

module s15
  real :: a(2), b(2)
  equivalence(a(2),b(1))
  !ERROR: 'a(3_8)' and 'a(1_8)' cannot have the same first storage unit
  equivalence(b(2),a(1))
end module
