! RUN: %python %S/test_errors.py %s %flang_fc1

function test_func(x) result(i)
  integer, pointer :: i
  real :: x
  x = func()
contains
  pure real function func()
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'i' is externally visible via 'i' and not definable in a pure subprogram
    i = 0
    func = 0.
  end function
end function
