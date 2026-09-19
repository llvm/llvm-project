! RUN: %check_flang_tidy %s bugprone-precision-loss %t
subroutine s
  real(8) :: i
  real(4) :: j
  i = 1.0
  j = 2.0
  j = i
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Possible loss of precision in implicit conversion (REAL(8) to REAL(4))
end subroutine s
