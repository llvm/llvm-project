! RUN: %B/test/Semantics/test_errors.sh %s %flang %t
! Test extension: RETURN from main program

return !ok
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
return 0
end
