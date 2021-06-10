! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
implicit none
allocatable :: x
integer :: x
!ERROR: No explicit type declared for 'y'
allocatable :: y
end
