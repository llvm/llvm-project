! RUN: %S/test_errors.sh %s %flang %t
implicit none
allocatable :: x
integer :: x
!ERROR: No explicit type declared for 'y'
allocatable :: y
end
