! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Test extension: RETURN from main program

return !ok
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
return 0
end
