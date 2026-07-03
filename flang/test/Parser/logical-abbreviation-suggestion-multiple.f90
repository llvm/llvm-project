! Every distinct .T./.F. abbreviation that causes a parse failure gets its own
! suggestion, not just the first one.  The operator form .A. parses as a defined
! operator and fails later in semantics rather than in the parser, so it gets no
! parse-time suggestion.

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

logical :: x
x = .T.
x = .F.
x = x .A. x
end

! CHECK-COUNT-2: This nonstandard logical abbreviation requires the '-flogical-abbreviations' option
! CHECK-NOT: This nonstandard logical abbreviation
