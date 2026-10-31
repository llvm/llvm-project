! RUN: %python %S/test_errors.py %s %flang_fc1
! DATA on components of a parameterized derived type instance is not yet
! supported.
program test
    type ut(n)
      integer, len :: n
      character(n) :: sar(2)
    end type
    type(ut(1)) pdt
    !ERROR: not yet implemented: DATA statement initialization of a component in a parameterized derived type instance
    data pdt%sar/'o','k'/
end program
