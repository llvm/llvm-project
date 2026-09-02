! RUN: %check_flang_tidy %s modernize-avoid-backspace-stmt %t
program backspace_test
  integer :: unit = 10, x

  open(unit, file="data.txt", action="readwrite")
  write(unit, *) 1, 2, 3

  backspace(unit)  ! This will trigger a warning
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Assign statements are not recommended

  read(unit, *) x
  close(unit)
end program backspace_test
