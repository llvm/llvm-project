! RUN: %check_flang_tidy %s modernize-avoid-data-constructs %t
program data_test
  integer :: array(3)

  data array /1, 2, 3/  ! This will trigger a warning
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Data statements are not recommended

  ! Better: array = [1, 2, 3]

  print *, array
end program data_test
