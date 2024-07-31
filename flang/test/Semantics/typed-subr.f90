! RUN: %python %S/test_errors.py %s %flang_fc1
! ERROR: SUBROUTINE prefix cannot specify a type
integer subroutine foo
end
