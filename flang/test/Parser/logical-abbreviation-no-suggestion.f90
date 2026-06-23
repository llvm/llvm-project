! Mitigation test: .f. is used here as a legitimate defined unary operator on
! one line, while an unrelated parse error occurs on a different line.  Because
! the failure is not on the line bearing the abbreviation, the
! -flogical-abbreviations suggestion must NOT be emitted.

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not='-flogical-abbreviations'

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
  r = .f. 4
  this is not valid
end program

! CHECK: error:
