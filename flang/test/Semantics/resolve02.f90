! RUN: %B/test/Semantics/test_errors.sh %s %flang %t
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
