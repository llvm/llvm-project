! RUN: %python %S/test_errors.py %s %flang_fc1
! References to functions whose results are plain procedures (invalid:
! a function result is either a variable or a procedure pointer, F2023
! 19.3.3), with the callee resolved through every FindFunctionResult path:
! interface block, dummy procedure, host association, ENTRY, type-bound
! procedure, generic, procedure-pointer entity, procedure-pointer component.

! Interface-block callee; reference is an output item.
subroutine s1()
  interface
    function g1() result(r)
      !ERROR: A function result may not be a procedure unless it is a procedure pointer
      procedure() :: r
    end function
  end interface
  !ERROR: Output item must not be a procedure
  print *, g1()
end

! Dummy-procedure callee.
subroutine s2(dp)
  interface
    function g2(x) result(r)
      real :: x
      !ERROR: A function result may not be a procedure unless it is a procedure pointer
      procedure() :: r
    end function
  end interface
  procedure(g2) :: dp
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(dp(1.0))
end

! Host-associated callee.
subroutine s3()
contains
  function g3(x) result(r)
    !ERROR: A function result may not be a procedure unless it is a procedure pointer
    procedure() :: r
    real :: x
  end function
  subroutine inner
    !ERROR: Actual argument for 'array=' may not be a procedure
    print *, size(g3(1.0))
  end subroutine
end

! ENTRY result.
function s4(x)
  real :: s4, x
  !ERROR: A function result may not be a procedure unless it is a procedure pointer
  procedure() :: j
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(e4(1.0))
  s4 = x
  return
  !ERROR: Result of ENTRY is not compatible with result of containing function
  entry e4(x) result(j)
end function

! Type-bound callee (ProcBindingDetails) and generic-resolved callee.
module m5
  type t5
  contains
    procedure, nopass :: tbp => g5
  end type
  interface gen5
    module procedure g5
  end interface
contains
  function g5(x) result(i)
    real, intent(in) :: x
    !ERROR: A function result may not be a procedure unless it is a procedure pointer
    procedure() :: i
  end function
end module

subroutine s5()
  use m5
  type(t5) :: x
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(x%tbp(1.0))
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(gen5(1.0))
end

! Procedure-pointer entity callee and proc-pointer component callee.
module m6
  interface
    function g6(y) result(r)
      !ERROR: A function result may not be a procedure unless it is a procedure pointer
      procedure() :: r
    end function
  end interface
  type t6
    procedure(g6), pointer, nopass :: pc
  end type
end module

subroutine s6()
  use m6
  procedure(g6), pointer :: pp
  type(t6) :: x
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(pp(1.0))
  !ERROR: Actual argument for 'array=' may not be a procedure
  print *, size(x%pc(1.0))
end
