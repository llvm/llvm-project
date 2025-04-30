! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test with only one procedure inside a submodule

MODULE m                          ! The ancestor module m
  INTEGER :: res
  INTERFACE
    MODULE SUBROUTINE sub2(arg2)
      INTEGER, intent(inout) :: arg2
    END SUBROUTINE
  END INTERFACE
END MODULE

SUBMODULE (m) n                   ! The descendant submodule n
  CONTAINS                        ! Module subprogram part
    MODULE procedure sub2         ! Definition of sub2 by separate module subprogram
      res = arg2 + 1
      if (res .ne. 100) then
        print *, "FAIL"
      else
        print *, "PASS"
      end if 
    END procedure sub2
END SUBMODULE

program test 
use m
implicit none
  integer :: h
  h = 99
  call sub2(h)
end program
