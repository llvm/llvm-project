! RUN: %check_flang_tidy %s bugprone-undeclared-procedure %t
subroutine s1
  call s2()
  ! CHECK-MESSAGES: :[[@LINE-1]]:8: warning: Implicit declaration of procedure 's2'
end subroutine s1
