! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests implemented for this standard:
!            Block Construct
! C1109

subroutine s5_c1109
  b1:block
  !ERROR: BLOCK construct name mismatch
  end block b2
end

