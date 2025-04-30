! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


subroutine weird(dummy)
  class(*), target :: dummy(:)
end subroutine weird

program main
  integer(kind=4), pointer :: myptr(:)
  integer :: j
  logical rslt(8), expect(8)
  interface
     subroutine weird(dummy)
       class(*), target :: dummy(:)
     end subroutine weird
  end interface
  allocate(myptr(4))
  myptr = [(111*j, j=1,4)]
  expect = .true.
  do j=1,4
     rslt(j) = (myptr(j) .eq. (111*j))
  enddo
  call weird(myptr)
  do j=1,4
     rslt(j+4) = (myptr(j) .eq. (111*j))
  enddo
  call check(rslt,expect,8)
end program main
