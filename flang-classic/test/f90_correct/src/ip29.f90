! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test that empty contained subprograms compile
!
PROGRAM test

   IMPLICIT NONE

   INTEGER :: m, n
   REAL :: x, y
   real result(1),expect(1)
   data expect/1/

   result(1) = 0
   CALL input
   CALL Write
   CALL inputs
   call check(result,expect,1)

CONTAINS

   SUBROUTINE Input
   END SUBROUTINE Input


   SUBROUTINE Write
      result(1) = 1
   END SUBROUTINE

   SUBROUTINE Inputs
   END SUBROUTINE Inputs
END PROGRAM test


