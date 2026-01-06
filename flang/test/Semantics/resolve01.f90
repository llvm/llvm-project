! RUN: %python %S/test_errors.py %s %flang_fc1
integer :: x
!ERROR: The type of 'x' has already been declared as INTEGER(4)
real :: x
integer(8) :: i
parameter(i=1,j=2,k=3)
integer :: j
!ERROR: The type of 'k' has already been implicitly declared as INTEGER(4)
real :: k
end
