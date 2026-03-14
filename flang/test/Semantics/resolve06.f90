! RUN: %python %S/test_errors.py %s %flang_fc1
implicit none
allocatable :: x
integer :: x
!ERROR: No explicit type declared for 'y'
allocatable :: y
end
