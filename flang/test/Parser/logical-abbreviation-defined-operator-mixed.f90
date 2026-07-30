! A valid defined-operator use of .f. must not swallow the suggestion for a
! later parse failure on the same spelling: the suggestion is emitted at the
! failing occurrence only, not at the valid one.

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not='This nonstandard logical abbreviation'

module m
  interface operator(.f.)
    module procedure neg
  end interface
contains
  pure integer function neg(a)
    integer, intent(in) :: a
    neg = -a
  end function
end module
program p
  use m
  integer :: r
  logical :: z
  r = .f.(4)
  z = .f.
end program

! CHECK: This nonstandard logical abbreviation requires the '-flogical-abbreviations' option
! CHECK-NEXT: z = .f.
