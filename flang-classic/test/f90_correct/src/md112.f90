! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! use only clause in multiple interface blocks

module mm
  integer, parameter :: kk = 4
end module mm

subroutine ss
  interface
    integer(kind=kk) function ff1(aa, bb)
      use mm, only : kk
      implicit none
      integer(kind=kk) :: aa, bb
    end function ff1
  end interface

  interface
    integer(kind=kk) function ff2(aa, bb)
      use mm, only : kk
      implicit none
      integer(kind=kk) :: aa, bb
    end function ff2
  end interface

  if (ff1(1,2) + ff2(3,4) .eq. 10) then
    print*, "PASS"
  else
    print*, "FAIL"
  endif
end subroutine ss

integer(kind=kk) function ff1(aa, bb)
  use mm, only : kk
  implicit none
  integer(kind=kk) :: aa, bb
  ff1 = aa + bb
end function ff1

integer(kind=kk) function ff2(aa, bb)
  use mm, only : kk
  implicit none
  integer(kind=kk) :: aa, bb
  ff2 = aa + bb
end function ff2

  call ss
end
