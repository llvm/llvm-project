! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
integer :: g(10)
f(i) = i + 1  ! statement function
g(i) = i + 2  ! mis-parsed array assignment
!ERROR: 'h' has not been declared as an array
h(i) = i + 3
end
