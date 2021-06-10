! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Tests implemented for this standard:
!            Block Construct
! C1109

subroutine s5_c1109
  b1:block
  !ERROR: BLOCK construct name mismatch
  end block b2
end

