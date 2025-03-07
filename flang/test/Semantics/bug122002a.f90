! RUN: %python %S/test_errors.py %s %flang_fc1
! ERROR: Missing initialization for parameter 'n'
! ERROR: Must be a scalar value, but is a rank-1 array
integer, parameter :: n(n)
end
