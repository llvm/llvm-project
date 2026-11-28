! RUN: %python %S/test_errors.py %s %flang_fc1
module m
contains
  subroutine s(a)
    complex, intent(inout) :: a
  end subroutine
end module
program p
  use m
  complex, target :: z
  complex, pointer :: q
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' is not definable
  !BECAUSE: '(z)' is not a variable or pointer
  call s(z**(1., 0.))
  !ERROR: Target associated with pointer 'q' must be a designator or a call to a pointer-valued function
  q => z**(1., 0.)
end program
