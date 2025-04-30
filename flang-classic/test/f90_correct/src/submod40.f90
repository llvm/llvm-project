! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

! C1548 (R1526) If MODULE appears in the prefix of a module subprogram, it shall
! have been declared to be a separate module procedure in the containing program
! unit or an ancestor of that program unit.
!

MODULE mod_test
  IMPLICIT NONE
  
  INTERFACE
    
    PURE INTEGER MODULE FUNCTION f1(i)
      INTEGER, INTENT(IN) :: i
    END FUNCTION f1
    
    PURE MODULE subroutine s1(i)
      INTEGER, INTENT(IN) :: i
    END subroutine s1

    MODULE SUBROUTINE sub1(arg1)
      INTEGER, intent(inout) :: arg1
    END SUBROUTINE

    MODULE SUBROUTINE sub2(arg2)
      INTEGER, intent(inout) :: arg2
    END SUBROUTINE

  END INTERFACE

END MODULE

SUBMODULE(mod_test) sub_mod
  IMPLICIT NONE
  
  CONTAINS
    
    PURE INTEGER MODULE FUNCTION f2(i) !{error "PGF90-S-1056-MODULE prefix is only allowed for subprograms that were declared as separate module procedures"}
      INTEGER, INTENT(IN) :: i
      f2 = i
    END FUNCTION f2

    PURE MODULE subroutine s1(i)
      INTEGER, INTENT(IN) :: i
    END subroutine s1

    MODULE PROCEDURE sub1
      arg1 = arg1 + 1 
    END PROCEDURE sub1

    MODULE PROCEDURE sub3 !{error "PGF90-S-1056-MODULE prefix is only allowed for subprograms that were declared as separate module procedures"} 

    END PROCEDURE sub3


END SUBMODULE sub_mod 

