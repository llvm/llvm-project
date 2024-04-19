! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  public i
  integer, private :: j
  !ERROR: The accessibility of 'i' has already been specified as PUBLIC
  private i
  !WARNING: The accessibility of 'j' has already been specified as PRIVATE
  private j
end

module m2
  interface operator(.foo.)
    module procedure ifoo
  end interface
  public :: operator(.foo.)
  !ERROR: The accessibility of 'OPERATOR(.foo.)' has already been specified as PUBLIC
  private :: operator(.foo.)
  interface operator(+)
    module procedure ifoo
  end interface
  public :: operator(+)
  !ERROR: The accessibility of 'OPERATOR(+)' has already been specified as PUBLIC
  private :: operator(+) , ifoo
contains
  integer function ifoo(x, y)
    logical, intent(in) :: x, y
  end
end module

module m3
  type t
  end type
  private :: operator(.lt.)
  interface operator(<)
    logical function lt(x, y)
      import t
      type(t), intent(in) :: x, y
    end function
  end interface
  !ERROR: The accessibility of 'OPERATOR(<)' has already been specified as PRIVATE
  public :: operator(<)
  interface operator(.gt.)
    logical function gt(x, y)
      import t
      type(t), intent(in) :: x, y
    end function
  end interface
  public :: operator(>)
  !ERROR: The accessibility of 'OPERATOR(.GT.)' has already been specified as PUBLIC
  private :: operator(.gt.)
end

module m4
  private
  type, public :: foo
  end type
  interface foo
    procedure fun
  end interface
 contains
  function fun
  end
end

subroutine s4
  !ERROR: 'fun' is PRIVATE in 'm4'
  use m4, only: foo, fun
  type(foo) x ! ok
  print *, foo() ! ok
end

module m5
  public
  type, private :: foo
  end type
  interface foo
    procedure fun
  end interface
 contains
  function fun
  end
end

subroutine s5
  !ERROR: 'foo' is PRIVATE in 'm5'
  use m5, only: foo, fun
  print *, fun() ! ok
end
