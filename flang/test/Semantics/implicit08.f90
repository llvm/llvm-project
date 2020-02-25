! RUN: %B/test/Semantics/test_errors.sh %s %flang %t
subroutine s1
  block
    !ERROR: IMPLICIT statement is not allowed in a BLOCK construct
    implicit logical(a)
  end block
end subroutine
