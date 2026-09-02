! RUN: %check_flang_tidy %s bugprone-missing-default-case %t
program case_test
  integer :: i = 2

  select case (i)  ! This will trigger a warning
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: SELECT CASE construct has no DEFAULT case
    case (1)
      print *, "One"
    case (2)
      print *, "Two"
    ! Missing: case default
  end select

  ! Correct form:
  select case (i)
    case (1)
      print *, "One"
    case (2)
      print *, "Two"
    case default
      print *, "Other"
  end select

end program case_test
