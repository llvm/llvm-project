! RUN: %check_flang_tidy %s performance-integer-power %t
subroutine s1(b, r)
  real, intent(in) :: b
  real, intent(out) :: r
  r = b ** 6.12E2
  ! CHECK-MESSAGES: :[[@LINE-1]]:12: warning: real exponent can be written as an integer literal
end subroutine s1

subroutine s2(b, r)
  real, intent(in) :: b
  real, intent(out) :: r
  r = b ** 6
end subroutine s2

subroutine s3(b, r)
  real, intent(in) :: b
  real, intent(out) :: r
  r = b ** 6.0
  ! CHECK-MESSAGES: :[[@LINE-1]]:12: warning: real exponent can be written as an integer literal
end subroutine s3

subroutine s4(b, r)
  real, intent(in) :: b
  real, intent(out) :: r
  r = b ** 6.12E1
end subroutine s4
