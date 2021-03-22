! RUN: %S/test_errors.sh %s %t %f18
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
  !ERROR: 'foo' may not be the name of both a generic interface and a procedure unless it is a specific procedure of the generic
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
  !ERROR: 'foo' is already declared in this scoping unit
  interface foo
    procedure :: foo
  end interface foo
contains
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
