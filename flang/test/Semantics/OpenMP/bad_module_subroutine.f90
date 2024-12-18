! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! Test that we don't crash on this code inside of openmp semantics checks

!ERROR: 'e' is a MODULE procedure which must be declared within a MODULE or SUBMODULE
impure elemental module subroutine e()
end subroutine
