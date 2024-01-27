! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  integer :: foo
  !Note: PGI, Intel, and GNU allow this; NAG and Sun do not
  !ERROR: 'foo' is already declared in this scoping unit
  interface foo
  end interface
end module

module m2
  interface s
  end interface
contains
  !WARNING: 's' should not be the name of both a generic interface and a procedure unless it is a specific procedure of the generic
  subroutine s
  end subroutine
end module

module m3
  ! This is okay: s is generic and specific
  interface s
    procedure s2
  end interface
  interface s
    procedure s
  end interface
contains
  subroutine s()
  end subroutine
  subroutine s2(x)
  end subroutine
end module

module m4a
  interface g
    procedure s_real
  end interface
contains
  subroutine s_real(x)
  end
end
module m4b
  interface g
    procedure s_int
  end interface
contains
  subroutine s_int(i)
  end
end
! Generic g should merge the two use-associated ones
subroutine s4
  use m4a
  use m4b
  call g(123)
  call g(1.2)
end

module m5a
  interface g
    procedure s_real
  end interface
contains
  subroutine s_real(x)
  end
end
module m5b
  interface gg
    procedure s_int
  end interface
contains
  subroutine s_int(i)
  end
end
! Generic g should merge the two use-associated ones
subroutine s5
  use m5a
  use m5b, g => gg
  call g(123)
  call g(1.2)
end

module m6a
  interface gg
    procedure sa
  end interface
contains
  subroutine sa(x)
  end
end
module m6b
  interface gg
    procedure sb
  end interface
contains
  subroutine sb(y)
  end
end
subroutine s6
  !ERROR: Generic 'g' may not have specific procedures 'sa' and 'sb' as their interfaces are not distinguishable
  use m6a, g => gg
  use m6b, g => gg
end

module m7a
  interface g
    procedure s1
  end interface
contains
  subroutine s1(x)
  end
end
module m7b
  interface g
    procedure s2
  end interface
contains
  subroutine s2(x, y)
  end
end
module m7c
  interface g
    procedure s3
  end interface
contains
  subroutine s3(x, y, z)
  end
end
! Merge the three use-associated generics
subroutine s7
  use m7a
  use m7b
  use m7c
  call g(1.0)
  call g(1.0, 2.0)
  call g(1.0, 2.0, 3.0)
end

module m8a
  interface g
    procedure s1
  end interface
contains
  subroutine s1(x)
  end
end
module m8b
  interface g
    procedure s2
  end interface
contains
  subroutine s2(x, y)
  end
end
module m8c
  integer :: g
end
! If merged generic conflicts with another USE, it is an error (if it is referenced)
subroutine s8
  use m8a
  use m8b
  use m8c
  !ERROR: Reference to 'g' is ambiguous
  g = 1
end

module m9a
  interface g
    module procedure g
  end interface
contains
  subroutine g()
  end
end module
module m9b
  interface g
    module procedure g
  end interface
contains
  subroutine g()
  end
end module
subroutine s9
  !ERROR: USE-associated generic 'g' may not have specific procedures 'g' and 'g' as their interfaces are not distinguishable
  use m9a
  use m9b
end

module m10a
  interface g
    module procedure s
  end interface
  private :: s
contains
  subroutine s(x)
    integer :: x
  end
end
module m10b
  use m10a
  !ERROR: Generic 'g' may not have specific procedures 's' and 's' as their interfaces are not distinguishable
  interface g
    module procedure s
  end interface
  private :: s
contains
  subroutine s(x)
    integer :: x
  end
end

module m12a
  interface ga
    module procedure sa
  end interface
contains
  subroutine sa(i)
  end
end
module m12b
  use m12a
  interface gb
    module procedure sb
  end interface
contains
  subroutine sb(x)
  end
end
module m12c
  use m12b, only: gc => gb
end
module m12d
  use m12a, only: g => ga
  use m12c, only: g => gc
  interface g
  end interface
end module

module m13a
 contains
  subroutine subr
  end subroutine
end module
module m13b
  use m13a
  interface subr
    module procedure subr
  end interface
end module
module m13c
  use m13a
  use m13b
 contains
  subroutine test
    call subr
  end subroutine
end module
module m13d
  use m13b
  use m13a
 contains
  subroutine test
    call subr
  end subroutine
end module

module m14a
  type :: foo
    integer :: n
  end type
end module
module m14b
  interface foo
    module procedure bar
  end interface
 contains
  real function bar(x)
    real, intent(in) :: x
    bar = x
  end function
end module
module m14c
  use m14a
  use m14b
  type(foo) :: x
end module
module m14d
  use m14a
  use m14b
  type(foo) :: x
 contains
  subroutine test
    real :: y
    y = foo(1.0)
    x = foo(2)
  end subroutine
end module
module m14e
  use m14b
  use m14a
  type(foo) :: x
 contains
  subroutine test
    real :: y
    y = foo(1.0)
    x = foo(2)
  end subroutine
end module

module m15a
  interface foo
    module procedure bar
  end interface
 contains
  subroutine bar
  end subroutine
end module
module m15b
  !ERROR: Cannot use-associate 'foo'; it is already declared in this scope
  use m15a
 contains
  subroutine foo
  end subroutine
end module
module m15c
 contains
  subroutine foo
  end subroutine
end module
module m15d
  use m15a
  use m15c
 contains
  subroutine test
    !ERROR: Reference to 'foo' is ambiguous
    call foo
  end subroutine
end module
