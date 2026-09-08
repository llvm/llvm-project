! RUN: %flang_fc1 -fsyntax-only %s
! Regression test: a substring of a character literal has the kind of the
! literal (F2023 9.4.1), so concatenating it with another literal of that
! kind is valid (F2023 10.1.5.3 requires the same kind on both operands).
! This must stay a -fsyntax-only test of a non-constant context: the value
! of a substring of a non-default-kind literal does not fold, so a constant
! context would be rejected, and lowering it is not implemented yet.
subroutine s()
  character(kind=4, len=3) :: t4
  character(kind=2, len=3) :: t2
  t4 = 4_"ab"(1:1) // 4_"cd"
  t2 = 2_"ab"(1:1) // 2_"cd"
end subroutine
