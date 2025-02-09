! RUN: %python %S/test_errors.py %s %flang_fc1
! ERROR: Must be a scalar value, but is a rank-1 array
! ERROR: Shape of initialized object 'a' must be constant
complex:: a(n) = SUM([1])
INTEGER, parameter :: n(2) = [2,2]
end
