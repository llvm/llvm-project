!RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
!CHECK:come-to-a-bad-end.f90:13:4: error: expected '('
!CHECK: in the context: statement function definition
!CHECK: in the context: SUBROUTINE subprogram
!CHECK:error: expected declaration construct
!CHECK:come-to-a-bad-end.f90:13:1: in the context: specification part
!CHECK: in the context: SUBROUTINE subprogram
!CHECK:error: end of file
!CHECK: in the context: SUBROUTINE subprogram
subroutine a
end
subroutine b
gnd
