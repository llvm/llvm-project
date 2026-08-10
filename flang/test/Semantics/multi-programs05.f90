! RUN: %python %S/test_errors.py %s %flang_fc1
! Test the restriction in 5.2.2

program m
end
!ERROR: A source file cannot contain more than one main program
program m2
end
