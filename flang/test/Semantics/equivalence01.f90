!RUN: not %flang_fc1 -pedantic %s 2>&1 | FileCheck %s
subroutine s1
  integer i, j
  real r(2)
  !CHECK: error: Equivalence set must have more than one object
  equivalence(i, j),(r(1))
end

subroutine s2
  integer i
  type t
    integer :: a
    integer :: b(10)
  end type
  type(t) :: x
  !CHECK: error: Derived type component 'x%a' is not allowed in an equivalence set
  equivalence(x%a, i)
  !CHECK: error: Derived type component 'x%b(2)' is not allowed in an equivalence set
  equivalence(i, x%b(2))
end

integer function f3(x)
  real x
  !CHECK: error: Dummy argument 'x' is not allowed in an equivalence set
  equivalence(i, x)
  !CHECK: error: Function result 'f3' is not allow in an equivalence set
  equivalence(f3, i)
end

subroutine s4
  integer :: y
  !CHECK: error: Pointer 'x' is not allowed in an equivalence set
  !CHECK: error: Allocatable variable 'y' is not allowed in an equivalence set
  equivalence(x, y)
  real, pointer :: x
  allocatable :: y
end

subroutine s5
  integer, parameter :: k = 123
  real :: x(10)
  real, save :: y[1:*]
  !CHECK: error: Coarray 'y' is not allowed in an equivalence set
  equivalence(x, y)
  !CHECK: error: Variable 'z' with BIND attribute is not allowed in an equivalence set
  equivalence(x, z)
  !CHECK: error: Variable 'z' with BIND attribute is not allowed in an equivalence set
  equivalence(x(2), z(3))
  real, bind(C) :: z(10)
  !CHECK: error: Named constant 'k' is not allowed in an equivalence set
  equivalence(x(2), k)
  !CHECK: error: Variable 'w' in common block with BIND attribute is not allowed in an equivalence set
  equivalence(x(10), w)
  logical :: w(10)
  bind(C, name="c") /c/
  common /c/ w
  integer, target :: u
  !CHECK: error: Variable 'u' with TARGET attribute is not allowed in an equivalence set
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
  !CHECK: error: Derived type object 'x1' with pointer ultimate component is not allowed in an equivalence set
  equivalence(x0, x1)
  !CHECK: error: Derived type object 'x2' with pointer ultimate component is not allowed in an equivalence set
  equivalence(x0, x2)
end

subroutine s7
  type t1
  end type
  real :: x0
  type(t1) :: x1
  !CHECK: error: Nonsequence derived type object 'x1' is not allowed in an equivalence set
  equivalence(x0, x1)
end

module m8
  real :: x
  real :: y(10)
end
subroutine s8
  use m8
  !CHECK: error: Use-associated variable 'x' is not allowed in an equivalence set
  equivalence(x, z)
  !CHECK: error: Use-associated variable 'y' is not allowed in an equivalence set
  equivalence(y(1), z)
end

subroutine s9
  character(10) :: c
  real :: d(10)
  integer, parameter :: n = 2
  integer :: i, j
  !CHECK: error: Substring with nonconstant bound 'n+j' is not allowed in an equivalence set
  equivalence(c(n+1:n+j), i)
  !CHECK: error: Substring with zero length is not allowed in an equivalence set
  equivalence(c(n:1), i)
  !CHECK: error: Array with nonconstant subscript 'j-1' is not allowed in an equivalence set
  equivalence(d(j-1), i)
  !CHECK: error: Array section 'd(1:n)' is not allowed in an equivalence set
  equivalence(d(1:n), i)
  character(4) :: a(10)
  equivalence(c, a(10)(1:2))
  !CHECK: error: 'a(10_8)(2_8:2_8)' and 'a(10_8)(1_8:1_8)' cannot have the same first storage unit
  equivalence(c, a(10)(2:3))
end

subroutine s10
  integer, parameter :: i(4) = [1, 2, 3, 4]
  real :: x(10)
  real :: y(4)
  !CHECK: error: Array with vector subscript 'i' is not allowed in an equivalence set
  equivalence(x(i), y)
end

subroutine s11(n)
  integer :: n
  real :: x(n), y
  !CHECK: error: Automatic object 'x' is not allowed in an equivalence set
  equivalence(x(1), y)
end

module s12
  real, protected :: a
  integer :: b
  !CHECK: error: Equivalence set cannot contain 'a' with PROTECTED attribute and 'b' without
  equivalence(a, b)
  !CHECK: error: Equivalence set cannot contain 'a' with PROTECTED attribute and 'b' without
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
  !CHECK: nonstandard: Equivalence set contains 'a' that is numeric sequence type and 'b' that is character
  equivalence(a, b)
  !CHECK: nonstandard: Equivalence set contains 'c' that is a default numeric sequence type and 'a' that is numeric with non-default kind
  equivalence(c, a)
  double precision :: d
  double complex :: e
  !OK: d and e are considered to be a default kind numeric type
  equivalence(c, d, e)
  type :: t3
    sequence
    real :: x
    character :: ch
  end type t3
  type(t3) :: s, r
  type :: t4
    sequence
    character :: ch
    real :: x
  end type t4
  type(t4) :: t
  !CHECK: nonstandard: Equivalence set contains 's' and 'r' with same type that is neither numeric nor character sequence type
  equivalence(s, r)
  !CHECK: error: Equivalence set cannot contain 's' and 't' with distinct types that are not both numeric or character sequence types
  equivalence(s, t)
end

module s14
  real :: a(10), b, c, d
  !CHECK: error: 'a(2_8)' and 'a(1_8)' cannot have the same first storage unit
  equivalence(a(1), a(2))
  equivalence(b, a(3))
  !CHECK: error: 'a(4_8)' and 'a(3_8)' cannot have the same first storage unit
  equivalence(a(4), b)
  equivalence(c, a(5))
  !CHECK: error: 'a(6_8)' and 'a(5_8)' cannot have the same first storage unit
  equivalence(a(6), d)
  equivalence(c, d)
end

module s15
  real :: a(2), b(2)
  equivalence(a(2),b(1))
  !CHECK: error: 'a(3_8)' and 'a(1_8)' cannot have the same first storage unit
  equivalence(b(2),a(1))
end module

subroutine s16

  integer var, dupName

  ! There should be no error message for the following
  equivalence (dupName, var)

  interface
    subroutine interfaceSub (dupName)
      integer dupName
    end subroutine interfaceSub
  end interface

end subroutine s16

module m17
  real :: dupName
contains
  real function f17a()
    implicit none
    real :: y
    !CHECK: error: No explicit type declared for 'dupname'
    equivalence (dupName, y)
  end function f17a
  real function f17b()
    real :: y
    ! The following implicitly declares an object called "dupName" local to
    ! the function f17b().  OK since there's no "implicit none
    equivalence (dupName, y)
  end function f17b
end module m17

module m18
  ! Regression test: don't loop when checking mutually-referencing types
  type t1
    sequence
    type (t2), pointer :: p
  end type
  type t2
    sequence
    type (t1), pointer :: p
  end type
  type(t1) x
  common x
end

subroutine s19
  entry e19
  !ERROR: 'e19' in equivalence set is not a data object
  equivalence (e19, j)
  !ERROR: 'e20' in equivalence set is not a data object
  equivalence (e20, j)
  entry e20
end
