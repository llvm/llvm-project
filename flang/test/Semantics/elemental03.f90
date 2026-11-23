!RUN: %python %S/test_errors.py %s %flang_fc1
module m
 contains
  elemental real function f(x)
    real, intent(in) :: x
    f = x
  end
  subroutine s(a)
    real a(..)
    !ERROR: Assumed-rank array 'a' may not be used as an argument to an elemental procedure
    print *, f(a)
  end
end
