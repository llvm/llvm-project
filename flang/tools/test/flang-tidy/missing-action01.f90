! RUN: %check_flang_tidy %s bugprone-missing-action %t
program open_test
  integer :: unit = 10

  open(10, file="data.txt")  ! This will trigger a warning for missing ACTION
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: File unit number is a constant literal
  ! CHECK-MESSAGES: :[[@LINE-2]]:3: warning: ACTION specifier is missing

  open(unit, file="data2.txt")  ! This will trigger a warning for constant unit
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ACTION specifier is missing

  close(10)
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: File unit number is a constant literal
end program open_test
