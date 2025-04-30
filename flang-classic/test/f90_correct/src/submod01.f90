! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Submodule test with only one suroutine test inside a submodule
!
MODULE m
  INTEGER :: res                          ! The ancestor module m
  INTERFACE
    MODULE SUBROUTINE sub1(arg1)          ! Module procedure interface body for sub1
      INTEGER, intent(inout) :: arg1
    END SUBROUTINE
  END INTERFACE
END MODULE

SUBMODULE (m) n                           ! The descendant submodule n
  CONTAINS                                ! Module subprogram part
    MODULE SUBROUTINE sub1(arg1)          ! Definition of sub1 by subroutine subprogram
      INTEGER, intent(inout) :: arg1
      arg1 = arg1 + 1
    END SUBROUTINE sub1
END SUBMODULE

program test 
use m
implicit none
  integer :: k, h
  k = 99
  call sub1(k)
  if (k .ne. 100) then
    print *, "FAIL"
  else
    print *, "PASS"
  end if 
end program test
