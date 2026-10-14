!RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type dt
  end type
 contains
  subroutine s(p)
    !ERROR: 'dt' must be an abstract interface or a procedure with an explicit interface
    procedure(dt), pointer :: p
  end
end
