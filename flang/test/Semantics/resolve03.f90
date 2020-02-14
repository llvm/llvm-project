! RUN: %S/test_errors.sh %s %flang %t
implicit none
integer :: x
!ERROR: No explicit type declared for 'y'
y = x
end
