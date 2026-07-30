! Each failing occurrence of a logical abbreviation gets its own suggestion,
! even when the spellings are identical: the recorded occurrences are keyed by
! source position, not by spelled text.

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not='This nonstandard logical abbreviation'

logical :: x, y
x = .F.
y = .F.
end

! CHECK: This nonstandard logical abbreviation requires the '-flogical-abbreviations' option
! CHECK-NEXT: x = .F.
! CHECK: This nonstandard logical abbreviation requires the '-flogical-abbreviations' option
! CHECK-NEXT: y = .F.
