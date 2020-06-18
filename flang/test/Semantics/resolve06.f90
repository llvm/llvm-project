! RUN: %S/test_errors.sh %s %t %f18
implicit none
allocatable :: x
integer :: x
!ERROR: No explicit type declared for 'y'
allocatable :: y
end
