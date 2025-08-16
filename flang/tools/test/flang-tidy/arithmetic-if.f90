! RUN: %check_flang_tidy %s bugprone-arithmetic-if %t
subroutine s
  real :: x = -1.0
  if (x) 10, 20, 30
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Arithmetic if statements are not recommended
10 print *, "X is negative"
   stop
20 print *, "X is zero"
   stop
30 print *, "X is positive"
   stop
end subroutine s
