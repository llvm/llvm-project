! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test with 3 levels of inheritance of submodules
!

MODULE m
  INTEGER :: res                      ! The ancestor module m
  INTERFACE
    MODULE SUBROUTINE sub1(arg1)      ! Module procedure interface body for sub1
      INTEGER, intent(inout) :: arg1
    END SUBROUTINE

    MODULE SUBROUTINE sub2(arg2)      ! Module procedure interface body for sub1
      INTEGER, intent(inout) :: arg2
    END SUBROUTINE
  END INTERFACE
END MODULE

SUBMODULE (m) n                       ! The descendant submodule n
  CONTAINS                            ! Module subprogram part
    MODULE SUBROUTINE sub1(arg1)      ! Definition of sub1 by subroutine subprogram
      INTEGER, intent(inout) :: arg1
      res = arg1 + 1
    END SUBROUTINE sub1
END SUBMODULE

SUBMODULE (m:n) k                     ! The descendant submodule k
  CONTAINS                            ! Module subprogram part
    MODULE SUBROUTINE sub2(arg2)      ! Definition of sub1 by subroutine subprogram
      INTEGER, intent(inout) :: arg2
      res = arg2 + 1
    END SUBROUTINE sub2
END SUBMODULE

program test 
use m
implicit none
  integer :: k, h
  k = 99
  call sub1(k)
  if (res .ne. 100) then 
    print *, "FAIL"
  else
    print *, "PASS"
  end if
  call sub2(k)
  if (res .ne. 100) then
    print *, "FAIL"
  else
    print *, "PASS"
  end if
end program test
