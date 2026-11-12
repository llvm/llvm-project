! RUN: %python %S/test_errors.py %s %flang_fc1
module acc_declare_test
! ERROR: Initialization expression cannot be converted to declared type of 'ifcondition' from LOGICAL(4)
 integer(16), parameter :: ifCondition = .FALSE.
end module
