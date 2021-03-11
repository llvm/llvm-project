! Ensure the end-of-line character and anything that follows after in a macro definition (-D) is ignored.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: printf -- "-DX=A\\\\\nTHIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT\n" | xargs %flang-new -E %s  2>&1 | FileCheck --strict-whitespace --match-full-lines %s

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: printf -- "-DX=A\\\\\nTHIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT\n" | xargs %flang-new -fc1 -E %s  2>&1 | FileCheck --strict-whitespace --match-full-lines %s

!-------------------------------
! EXPECTED OUTPUT FOR MACRO 'X'
!-------------------------------
! CHECK:start a end
! CHECK-NOT:THIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT
! CHECK-NOT:this_should_not_exist_in_the_output

START X END