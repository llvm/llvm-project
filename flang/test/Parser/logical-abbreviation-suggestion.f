! Fixed-form counterpart: the .F. abbreviation in an assignment is first
! misparsed as a statement function (so the error points at the '='), but the
! suggestion is still anchored at the abbreviation and offered to the user.

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fsyntax-only -flogical-abbreviations %s

      logical flag
      flag = .F.
      end

! CHECK: error: expected '('
! CHECK: This nonstandard logical abbreviation requires the '-flogical-abbreviations' option
