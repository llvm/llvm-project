! RUN: %S/test_errors.sh %s %t %f18
implicit none
integer :: x
!ERROR: No explicit type declared for 'y'
y = x
end
