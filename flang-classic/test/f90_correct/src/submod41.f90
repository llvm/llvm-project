! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

! C1547 (R1526) MODULE shall appear only in the function-stmt or subroutine-stmt
! of a module subprogram or of a nonabstract interface body that is declared in
! the scoping unit of a module or submodule.

PROGRAM test
  IMPLICIT NONE
  
INTERFACE
    PURE INTEGER MODULE FUNCTION f1(i) !{error "PGF90-S-0310-MODULE prefix allowed only within a module or submodule"}
      INTEGER, INTENT(IN) :: i
    END FUNCTION f1
END INTERFACE

END PROGRAM test
