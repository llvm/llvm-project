! RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  implicit none
contains
  subroutine foo(x)
    real :: x
  end subroutine
end module

!Note: PGI, Intel, GNU, and NAG allow this; Sun does not
module m2
  use m1
  implicit none
  !WARNING: 'foo' should not be the name of both a generic interface and a procedure unless it is a specific procedure of the generic
  interface foo
    module procedure s
  end interface
contains
  subroutine s(i)
    integer :: i
  end subroutine
end module

subroutine foo
  !ERROR: Cannot use-associate 'foo'; it is already declared in this scope
  use m1
end

subroutine bar
  !ERROR: Cannot use-associate 'bar'; it is already declared in this scope
  use m1, bar => foo
end

!OK to use-associate a type with the same name as a generic
module m3a
  type :: foo
  end type
end
module m3b
  use m3a
  interface foo
  end interface
end

! Can't have derived type and function with same name
module m4a
  type :: foo
  end type
contains
  !ERROR: 'foo' is already declared in this scoping unit
  function foo(x)
  end
end
! Even if there is also a generic interface of that name
module m4b
  type :: foo
  end type
  interface foo
    procedure :: foo
  end interface foo
contains
  !ERROR: 'foo' is already declared in this scoping unit
  function foo(x)
  end
end
module m4c
  type :: foo
  end type
  interface foo
    !ERROR: 'foo' is already declared in this scoping unit
    real function foo()
    end function foo
  end interface foo
end

! Use associating a name that is a generic and a derived type
module m5a
  interface g
  end interface
  type g
  end type
end module
module m5b
  use m5a
  interface g
    procedure f
  end interface
  type(g) :: x
contains
  function f(i)
  end function
end module
subroutine s5
  use m5b
  type(g) :: y
end

module m6
  real :: f6
  interface g6
  !ERROR: 'f6' is already declared in this scoping unit
    real function f6()
    end function f6
  end interface g6
end module m6

module m7
  integer :: f7
  interface g7
    !ERROR: 'f7' is already declared in this scoping unit
    real function f7()
    end function f7
  end interface g7
end module m7

module m8
  real :: f8
  interface g8
    !ERROR: 'f8' is already declared in this scoping unit
    subroutine f8()
    end subroutine f8
  end interface g8
end module m8

module m9
  type f9
  end type f9
  interface f9
    real function f9()
    end function f9
  end interface f9
contains
  !ERROR: 'f9' is already declared in this scoping unit
  function f9(x)
  end function f9
end module m9

module m10
  type :: t10
  end type t10
  interface f10
    function f10()
    end function f10
  end interface f10
contains
  !ERROR: 'f10' is already declared in this scoping unit
  function f10(x)
  end function f10
end module m10

module m11
  type :: t11
  end type t11
  interface i11
    function f11()
    end function f11
  end interface i11
contains
  !ERROR: 'f11' is already declared in this scoping unit
  function f11(x)
  end function f11
end module m11

module m12
  interface f12
    function f12()
    end function f12
  end interface f12
contains
  !ERROR: 'f12' is already declared in this scoping unit
  function f12(x)
  end function f12
end module m12

module m13
  interface f13
    function f13()
    end function f13
  end interface f13
contains
  !ERROR: 'f13' is already declared in this scoping unit
  function f13()
  end function f13
end module m13

! Not an error
module m14
  interface gen1
    module procedure s
  end interface
  generic :: gen2 => s
 contains
  subroutine s(x)
    integer(1) :: x
  end subroutine s
end module m14
module m15
  use m14
  interface gen1
    module procedure gen1
  end interface
  generic :: gen2 => gen2
 contains
  subroutine gen1(x)
    integer(2) :: x
  end subroutine gen1
  subroutine gen2(x)
    integer(4) :: x
  end subroutine gen2
end module m15

module m15a
  interface foo
    module procedure foo
  end interface
 contains
  function foo()
  end
end

module m15b
  interface foo
    module procedure foo
  end interface
 contains
  function foo(x)
  end
end

subroutine test15
  use m15a
  use m15b ! ok
end


module m16a
  type foo
    integer j
  end type
  interface foo
    module procedure bar
  end interface
 contains
  function bar(j)
  end
end

module m16b
  type foo
    integer j, k
  end type
  interface foo
    module procedure bar
  end interface
 contains
  function bar(x,y)
  end
end

subroutine test16
  use m16a
  use m16b ! ok
end

subroutine test17
  use m15a
  use m16a ! ok
end

subroutine test18
  use m16a
  use m15a ! ok
end

module m21
  type foo
    integer a
  end type
  interface foo
    module procedure f1
  end interface
 contains
  function f1(a)
    f1 = a
  end
end

module m22
  type foo
    real b
  end type
  interface foo
    module procedure f2
  end interface
 contains
  function f2(a,b)
    f2 = a + b
  end
end

module m23
  interface foo
    module procedure foo
    module procedure f3
  end interface
 contains
  function foo()
    foo = 0.
  end
  function f3(a,b,c)
    f3 = a + b + c
  end
end

module m24
  interface foo
    module procedure foo
    module procedure f4
  end interface
 contains
  function foo(a)
    foo = a
  end
  function f4(a,b,c,d)
    f4 = a + b + c +d
  end
end

subroutine s_21_22_a
  use m21
  use m22
  print *, foo(1.) ! Intel error
  print *, foo(1.,2.) ! Intel error
end

subroutine s_21_22_b
  use m21
  use m22
  !ERROR: 'foo' is not a derived type
  type(foo) x ! definite error: GNU and Intel catch
end

subroutine s_21_23
  use m21
  use m23
  type(foo) x ! Intel and NAG error
  print *, foo(1.) ! Intel error
  print *, foo(1.,2.,3.) ! Intel error
  call ext(foo) ! GNU and Intel error
end

subroutine s_22_23
  use m22
  use m23
  type(foo) x ! Intel and NAG error
  print *, foo(1.,2.) ! Intel error
  print *, foo(1.,2.,3.) ! Intel error
  call ext(foo) ! Intel error
end

subroutine s_23_24
  use m23
  use m24
  print *, foo(1.,2.,3.) ! NAG error
  print *, foo(1.,2.,3.,4.) ! XLF error
  !ERROR: 'foo' is not a specific procedure
  call ext(foo) ! definite error
end
