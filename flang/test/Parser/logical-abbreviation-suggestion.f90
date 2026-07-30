! Verify that a parse failure caused by the disabled .F. logical abbreviation
! still reports the usual errors and additionally suggests the
! -flogical-abbreviations option, and that enabling the option compiles cleanly.

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fsyntax-only -flogical-abbreviations %s

logical :: x
x = .F.
end

! CHECK: error: expected '('
! CHECK: This nonstandard logical abbreviation requires the '-flogical-abbreviations' option
