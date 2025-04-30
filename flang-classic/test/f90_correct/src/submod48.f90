! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

MODULE m
  INTEGER :: res
  INTERFACE
    PURE MODULE SUBROUTINE sub1(arg1)
      INTEGER, intent(inout) :: arg1
    END SUBROUTINE
  END INTERFACE
END MODULE

SUBMODULE (m) n
  CONTAINS
    MODULE SUBROUTINE sub1(arg1) !{error "PGF90-S-1060-The PURE function prefix of the definition and declaration of subprogram sub1 must match"}
      REAL, intent(inout) :: arg1
      print *, arg1
    END SUBROUTINE sub1
END SUBMODULE

program test
use m
integer :: a = 10
call sub1(a)


end 

