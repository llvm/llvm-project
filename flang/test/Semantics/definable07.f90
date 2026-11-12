! RUN: %python %S/test_errors.py %s %flang_fc1
integer, parameter :: j = 5
real a(5)
!ERROR: 'j' is not a variable
read *, (a(j), j=1, 5)
!ERROR: 'j' is not a variable
print *, (a(j), j=1, 5)
end
