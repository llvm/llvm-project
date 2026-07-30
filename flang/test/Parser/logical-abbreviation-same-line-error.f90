! Known limitation of the line-granularity heuristic: a valid defined-operator
! use of .f. that shares its source line with an unrelated parse error still
! receives the -flogical-abbreviations suggestion, because suggestions are tied
! to failing source lines rather than to the failing token itself.

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

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
  r = .f. 4 garbage here
end program

! CHECK: error: Could not parse
! CHECK: This nonstandard logical abbreviation requires the '-flogical-abbreviations' option
! CHECK-NEXT: r = .f. 4 garbage here
