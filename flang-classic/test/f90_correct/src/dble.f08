!                                                        
! Part of the LLVM Project, under the Apache License v2.0
! See https://llvm.org/LICENSE.txt for license informatio
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check if dble function take quadruple precision properly

program main
  use check_mod
  real(8) :: A(5) = [dble(-1.79E308_16), &
  dble(-2.23E-308_16), dble(0.0_16), &
  dble(2.23E-308_16), dble(1.79E308_16)]

  real(8), parameter :: B(5) = [dble(-1.79E308_16), &
  dble(-2.23E-308_16), dble(0.0_16), &
  dble(2.23E-308_16), dble(1.79E308_16)]

  real(8), parameter :: expect(10) = &
   [-1.7900000000000000E+308_8, -2.2300000000000001E-308_8,&
    0.0000000000000000_8, 2.2300000000000001E-308_8,&
    1.7900000000000000E+308_8, -1.7900000000000000E+308_8, &
    -2.2300000000000001E-308_8, 0.0000000000000000_8, &
    2.2300000000000001E-308_8, 1.7900000000000000E+308_8]

  real(8) :: result(10)
  integer :: i
  do i = 1,10
    if (i <= 5) then
      result(i) = A(i)
    else
      result(i) = B(i-5)
    end if
  end do

  call checkr8(result, expect, 10)
end program main
