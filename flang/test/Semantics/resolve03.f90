! RUN: %S/test_errors.sh %s %t %flang_fc1
implicit none
integer :: x
!ERROR: No explicit type declared for 'y'
y = x
end
