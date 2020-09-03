! RUN: %S/test_errors.sh %s %t %f18
subroutine s
  !ERROR: Declaration of 'x' conflicts with its use as internal procedure
  real :: x
contains
  subroutine x
  end
end

module m
  !ERROR: Declaration of 'x' conflicts with its use as module procedure
  real :: x
contains
  subroutine x
  end
end
