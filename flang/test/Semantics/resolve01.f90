! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
integer :: x
!ERROR: The type of 'x' has already been declared
real :: x
integer(8) :: i
parameter(i=1,j=2,k=3)
integer :: j
!ERROR: The type of 'k' has already been implicitly declared
real :: k
end
