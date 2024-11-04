! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! CompilerDirective with openmp tests

!ERROR: !DIR$ IGNORE_TKR directive must appear in a subroutine or function
!dir$ ignore_tkr

program main
end program main
