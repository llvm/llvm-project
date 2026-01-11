! Test that EXIT accepts INTEGER of any kind
! RUN: %flang_fc1 %s

program test_exit_any_integer
  implicit none
  integer(1) :: s1 = 1
  integer(2) :: s2 = 2
  integer(4) :: s4 = 4
  integer(8) :: s8 = 8

  call exit(s1)
  call exit(s2)
  call exit(s4)
  call exit(s8)

end program test_exit_any_integer