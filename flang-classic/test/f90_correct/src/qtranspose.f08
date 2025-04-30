! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test TRANSPOSE intrinsic with quad-precision arguments; also test
! the functionality of IEEE_NEXT_AFTER for quad-precision results

program main
  use ieee_arithmetic
  use check_mod
  real(16), dimension(3,3) :: src, expct
  do i = 1, 3
    do j = 1, 3
      src(i, j) = ieee_next_after(real(i**j, kind = 16), 0.0_16)
    enddo
  enddo
  expct(1, 1) = 0.999999999999999999999999999999999904_16  
  expct(2, 1) = 0.999999999999999999999999999999999904_16 
  expct(3, 1) = 0.999999999999999999999999999999999904_16
  expct(1, 2) = 1.99999999999999999999999999999999981_16
  expct(2, 2) = 3.99999999999999999999999999999999961_16
  expct(3, 2) = 7.99999999999999999999999999999999923_16
  expct(1, 3) = 2.99999999999999999999999999999999961_16
  expct(2, 3) = 8.99999999999999999999999999999999846_16
  expct(3, 3) = 26.9999999999999999999999999999999969_16

  src = transpose(src)
  call checkr16(src, expct, 9)
end
