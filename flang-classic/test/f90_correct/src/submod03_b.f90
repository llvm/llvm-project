! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test of using derived type inside procedures of a sudmodule
!

MODULE m1
  TYPE Base
    INTEGER :: a
  END TYPE
  
  INTERFACE
    MODULE SUBROUTINE sub1(b)     ! Module procedure interface body for sub1
      TYPE(Base), INTENT(IN) :: b
    END SUBROUTINE
  END INTERFACE
END MODULE

SUBMODULE (m1) m1sub
  CONTAINS
    MODULE PROCEDURE sub1         ! Implementation of sub1 declared in m1
      !PRINT *, "sub1", b
      if (b%a .ne. 11) then
        print *, "FAIL"
      else
        print *, "PASS"
      end if
    END PROCEDURE
END SUBMODULE

PROGRAM example
  USE m1
  implicit none
  CALL sub1(Base(11))
END PROGRAM
