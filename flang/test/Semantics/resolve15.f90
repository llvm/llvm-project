! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  real :: var
  interface i
    !ERROR: 'var' is not a subprogram
    procedure :: sub, var
    !ERROR: Procedure 'bad' not found
    procedure :: bad
  end interface
  interface operator(.foo.)
    !ERROR: 'var' is not a subprogram
    procedure :: var
    !ERROR: OPERATOR(.foo.) procedure 'sub' must be a function
    procedure :: sub
    !ERROR: Procedure 'bad' not found
    procedure :: bad
  end interface
contains
  subroutine sub
  end
end

subroutine s
  interface i
    !ERROR: 'sub' is not a module procedure
    module procedure :: sub
  end interface
  interface assignment(=)
    !ERROR: 'sub' is not a module procedure
    module procedure :: sub
  end interface
contains
  subroutine sub(x, y)
    real, intent(out) :: x
    logical, intent(in) :: y
  end
end
