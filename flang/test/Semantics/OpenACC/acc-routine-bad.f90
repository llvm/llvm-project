! RUN: %python %S/../test_errors.py %s %flang -fopenacc -Werror

module acc_routine_bad

contains
!$acc routine
!WARNING: OpenACC routine directive without name must be placed in a subroutine or function
subroutine acc_routine20()
end subroutine
end module
