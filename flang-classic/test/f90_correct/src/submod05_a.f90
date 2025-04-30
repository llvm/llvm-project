! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test of using with two procedures inside a submodule.
!

MODULE m                          ! The ancestor module m
  INTERFACE
    MODULE SUBROUTINE sub1(arg1)  ! Module procedure interface body for sub2
      INTEGER, intent(inout) :: arg1
    END SUBROUTINE

    MODULE SUBROUTINE sub2(arg2)  ! Module procedure interface body for sub2
      INTEGER, intent(inout) :: arg2
    END SUBROUTINE

  END INTERFACE
END MODULE

SUBMODULE (m) n                   ! The descendant submodule n
  CONTAINS                        ! Module subprogram part
    MODULE PROCEDURE sub1         ! Definition of sub2 by separate module subprogram
      arg1 = arg1 + 1
      if (arg1 .ne. 100) then
        print *, "FAIL"
      else 
        print *, "PASS"
      end if
    END PROCEDURE sub1

    MODULE PROCEDURE sub2         ! Definition of sub2 by separate module subprogram
      if (arg2 .ne. 100) then
        print *, "FAIL"
      else 
        print *, "PASS"
      end if
     
    END PROCEDURE sub2

END SUBMODULE

program test 
use m
implicit none
  integer :: h
  h = 99
  call sub1(h)
  call sub2(h)
end program
