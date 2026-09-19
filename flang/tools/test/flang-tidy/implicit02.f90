! RUN: %check_flang_tidy %s bugprone-implicit-declaration %t
subroutine s(a, n)
  real, intent(in) :: a(n)
  ! CHECK-MESSAGES: :[[@LINE-2]]:17: warning: Implicit declaration of symbol 'n'
end
