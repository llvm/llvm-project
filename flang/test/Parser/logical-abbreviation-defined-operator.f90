! A program that legitimately uses .f. as a defined unary operator parses and
! compiles cleanly with the default (LogicalAbbreviations disabled).  The
! abbreviation spelling is recorded by the parser, but because the parse
! succeeds no -flogical-abbreviations suggestion is emitted.

! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --allow-empty --implicit-check-not='-flogical-abbreviations'

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
  print *, r
end program
