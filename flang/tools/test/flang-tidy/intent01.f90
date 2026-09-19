! RUN: %check_flang_tidy %s bugprone-unused-intent %t
subroutine s(a, b)
  integer, intent(inout) :: a, b
  ! CHECK-MESSAGES: :[[@LINE-1]]:32: warning: Dummy argument 'b' with intent(inout) is never written to, consider changing to intent(in)
  a = a + 1 - b
end subroutine s
