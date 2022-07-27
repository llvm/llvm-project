! Ensure the end-of-line character and anything that follows after in a macro definition (-D) is ignored.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: printf -- "-DX=A\\\\\nTHIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT\n" | xargs %flang -E -P %s  2>&1 | FileCheck --strict-whitespace --match-full-lines %s

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang_fc1)
!-----------------------------------------
! RUN: printf -- "-DX=A\\\\\nTHIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT\n" | xargs %flang_fc1 -E -P %s  2>&1 | FileCheck --strict-whitespace --match-full-lines %s

! CHECK:       START A END
! CHECK-NOT:THIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT
! CHECK-NOT:this_should_not_exist_in_the_output

       START X END
