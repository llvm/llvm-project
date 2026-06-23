! The .T. spelling of .TRUE. is handled like .F.: the usual parse error plus a
! suggestion to enable -flogical-abbreviations.

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fsyntax-only -flogical-abbreviations %s

program p
  logical :: x
  x = .T.
end program

! CHECK: error: expected '('
! CHECK: This nonstandard logical abbreviation requires the '-flogical-abbreviations' option
