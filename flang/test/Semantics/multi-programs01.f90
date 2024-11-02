! RUN: %python %S/test_errors.py %s %flang_fc1
! Test the restriction in 5.2.2

end
!ERROR: A source file cannot contain more than one main program
end
