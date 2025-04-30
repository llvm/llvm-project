! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check DO loops with real*16

program test
  use check_mod
  real(16) :: r, result(3), expect(3)
  integer :: i
  do r = 1, 3
    result = r
  enddo
  
  do i = 1, 3
    expect = i
  enddo
  call checkr16(result, expect, 3)
 
end program
