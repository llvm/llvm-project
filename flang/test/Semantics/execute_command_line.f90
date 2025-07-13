! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Tests for the EXECUTE_COMMAND_LINE intrinsics

subroutine bad_kind_error(command, exitVal, cmdVal)
CHARACTER(30) :: command
INTEGER(KIND=2) :: exitVal
INTEGER(KIND=1) :: cmdVal
!ERROR: Actual argument for 'exitstat=' has bad type or kind 'INTEGER(2)'
call execute_command_line(command, exitstat=exitVal)

!ERROR: Actual argument for 'cmdstat=' has bad type or kind 'INTEGER(1)'
call execute_command_line(command, cmdstat=cmdVal)
end subroutine bad_kind_error

subroutine good_kind_equal(command, exitVal, cmdVal)
CHARACTER(30) :: command
INTEGER(KIND=4) :: exitVal
INTEGER(KIND=2) :: cmdVal
call execute_command_line(command, exitstat=exitVal)
call execute_command_line(command, cmdstat=cmdVal)
end subroutine good_kind_equal

subroutine good_kind_greater(command, exitVal, cmdVal)
CHARACTER(30) :: command
INTEGER(KIND=8) :: exitVal
INTEGER(KIND=4) :: cmdVal
call execute_command_line(command, exitstat=exitVal)
call execute_command_line(command, cmdstat=cmdVal)
end subroutine good_kind_greater
