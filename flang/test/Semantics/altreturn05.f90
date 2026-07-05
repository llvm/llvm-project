! RUN: %python %S/test_errors.py %s %flang_fc1
! Test extension: RETURN from main program

return !ok
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
return 0
end
