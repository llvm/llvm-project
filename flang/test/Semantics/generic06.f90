! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  !ERROR: Specific procedure 'sin' of generic interface 'yintercept' may not be INTRINSIC
  intrinsic sin
  interface yIntercept
    procedure sin
  end interface
  !ERROR: Specific procedure 'cos' of generic interface 'xintercept' may not be INTRINSIC
  intrinsic cos
  generic :: xIntercept => cos
end module

subroutine foo
  interface slope
    procedure tan
  end interface
  !ERROR: Specific procedure 'tan' of generic interface 'slope' may not be a statement function
  tan(x) = sin(x) / cos(x)
end subroutine

