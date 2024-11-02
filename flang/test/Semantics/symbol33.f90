! RUN: %python %S/test_symbols.py %s %flang_fc1
! Ensure that a misparsed function reference that turns out to be an
! array element reference still applies implicit typing, &c.
!DEF: /subr (Subroutine) Subprogram
subroutine subr
  !DEF: /subr/moo (Implicit) ObjectEntity INTEGER(4)
  common //moo(1)
  !DEF: /subr/a ObjectEntity REAL(4)
  !REF: /subr/moo
  real a(moo(1))
end subroutine
