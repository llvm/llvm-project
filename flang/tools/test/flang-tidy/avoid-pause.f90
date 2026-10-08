! RUN: %check_flang_tidy %s modernize-avoid-pause-stmt %t
program pause_test
  print *, "Processing data..."

  pause  ! This will trigger a warning
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Pause statements are not recommended

  ! Better: use READ or other interactive techniques instead

  print *, "Continuing execution..."
end program pause_test
