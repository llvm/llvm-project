! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! Check out-of-range subscripts
real a(10)
integer, parameter :: n(2) = [1, 2]
!ERROR: DATA statement designator 'a(0_8)' is out of range
!ERROR: DATA statement designator 'a(11_8)' is out of range
data a(0)/0./, a(10+1)/0./
!ERROR: Subscript 0 is less than lower bound 1 for dimension 1 of array
print *, a(0)
!ERROR: Subscript 0 is less than lower bound 1 for dimension 1 of array
print *, a(1-1)
!ERROR: Subscript 11 is greater than upper bound 10 for dimension 1 of array
print *, a(11)
!ERROR: Subscript 11 is greater than upper bound 10 for dimension 1 of array
print *, a(10+1)
!ERROR: Subscript value (0) is out of range on dimension 1 in reference to a constant array value
print *, n(0)
!ERROR: Subscript value (3) is out of range on dimension 1 in reference to a constant array value
print *, n(4-1)
print *, a(1:12:3) ! ok
!ERROR: Subscript 13 is greater than upper bound 10 for dimension 1 of array
print *, a(1:13:3)
print *, a(10:-1:-3) ! ok
!ERROR: Subscript -2 is less than lower bound 1 for dimension 1 of array
print *, a(10:-2:-3)
print *, a(-1:-2) ! empty section is ok
print *, a(0:11:-1) ! empty section is ok
end

