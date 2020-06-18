! RUN: %S/test_errors.sh %s %t %f18
subroutine s1
  block
    !ERROR: IMPLICIT statement is not allowed in a BLOCK construct
    implicit logical(a)
  end block
end subroutine
