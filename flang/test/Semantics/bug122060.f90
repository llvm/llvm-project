! RUN: %python %S/test_errors.py %s %flang_fc1
integer, parameter :: a = -10
! ERROR: Assignment to constant 'a' is not allowed
! ERROR: 'shape=' argument ([INTEGER(4)::-10_4]) must not have a negative extent
a = b() - reshape([c], [a])
END
