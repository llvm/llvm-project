! RUN: %check_flang_tidy %s bugprone-arithmetic-goto %t
subroutine s(i)
  integer, intent(in) :: i
  goto (10, 20, 30), i
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Arithmetic goto statements are not recommended

  10 continue
  20 continue
  30 continue
end subroutine s
