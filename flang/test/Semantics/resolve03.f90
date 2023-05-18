! RUN: %python %S/test_errors.py %s %flang_fc1
implicit none
integer :: x
!ERROR: No explicit type declared for 'y'
y = x
end
