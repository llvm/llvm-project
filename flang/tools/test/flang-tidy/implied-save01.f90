! RUN: %check_flang_tidy %s bugprone-implied-save %t
subroutine s
  integer :: counter = 0
  ! CHECK-MESSAGES: :[[@LINE-1]]:14: warning: Implicit SAVE on symbol 'counter'
  counter = counter + 1
  print *, "Called", counter, "times"
end subroutine s

subroutine explicit_save
  integer, save :: counter = 0  ! No warning - explicitly saved
  counter = counter + 1
  print *, "Called", counter, "times"
end subroutine explicit_save
