! RUN: %check_flang_tidy %s bugprone-implicit-declaration %t
subroutine s
  integer :: i, j
  i = 1
  j = 2
  k = i + j
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Implicit declaration of symbol 'k'
end subroutine s
