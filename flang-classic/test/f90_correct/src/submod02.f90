! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test with only one function inside a submodule.
!
MODULE m                                      ! The ancestor module m
  INTERFACE
    MODULE FUNCTION sub2(arg2) result (res)   ! Module procedure interface body for sub2
      INTEGER, intent(in) :: arg2
      INTEGER :: res
    END FUNCTION
  END INTERFACE
END MODULE

SUBMODULE (m) n                               ! The descendant submodule n
  CONTAINS                                    ! Module subprogram part
    MODULE FUNCTION sub2(arg2) result (res)   ! Definition of sub2 by separate module subprogram
      INTEGER, intent(in) :: arg2
      INTEGER :: res
      res = arg2 + 1
    END FUNCTION sub2
END SUBMODULE

program test 
use m
implicit none
  integer :: h
  integer :: haha
  h = 99
  haha = sub2(h)
  if (haha .ne. 100) then
    print *, "FAIL"
  else
    print *, "PASS"
  end if 
end program test
