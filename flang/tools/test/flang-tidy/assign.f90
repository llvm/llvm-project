! RUN: %check_flang_tidy %s modernize-avoid-assign-stmt %t
subroutine s
  integer :: i
  assign 9 to i
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Assign statements are not recommended    
  go to i
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Assigned Goto statements are not recommended
9 continue
end subroutine s
