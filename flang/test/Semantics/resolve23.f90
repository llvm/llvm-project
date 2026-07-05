! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type :: t
    real :: y
  end type
end module

use m
implicit type(t)(x)
z = x%y  !OK: x is type(t)
!ERROR: 'w' is not an object of derived type; it is implicitly typed
z = w%y
end
