! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine bad(a)
  real :: a(..)
  !ERROR: Selector must not be assumed-rank
  associate(x => a)
  end associate
end subroutine
