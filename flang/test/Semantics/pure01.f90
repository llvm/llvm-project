! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensure that an impure bound operator can't be called
! from a pure context.
module m
  type t
   contains
    procedure :: binding => func
    generic :: operator(.not.) => binding
  end type
 contains
  impure integer function func(x)
    class(t), intent(in) :: x
    func = 0
  end
  pure integer function test
    !ERROR: Procedure 'func' referenced in pure subprogram 'test' must be pure too
    test = .not. t()
  end
end
