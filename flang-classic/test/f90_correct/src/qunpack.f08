! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test UNPACK intrinsic with quad-precision arguments

program main
  use ieee_arithmetic
  real(16), dimension(3, 3, 3) :: filed, res, expct
  real(16) :: vector(9), r(3, 3, 3, 2), er(3, 3, 3, 2)
  logical, dimension(3, 3, 3) :: mas = .false.

  filed = 9999.88888888888888888_16
  vector(1) = huge(1.0_16)
  vector(2) = ieee_next_after(1.0_16,0.0_16)
  vector(3) = ieee_next_after(1.0_16,2.0_16)
  do i = 4, 9
  vector(i) = real(i**i, kind = 16)
  enddo
  do i = 1, 3
  mas(i,1,1) = .true.
  mas(3,2,i) = .true.
  enddo
  expct = 9999.88888888888888887999999999999986_16
  res = unpack(vector, mas, filed)
  call init(expct)
  if(all(res .eq. expct) .neqv. .true.) STOP 1
  expct = 0.0_16
  res = unpack(vector, mas, 0.0_16)
  call init(expct)
  if(all(res .eq. expct) .neqv. .true.) STOP 2
  print *,'PASS'

end

subroutine init(expct)
  real(16), dimension(3, 3, 3) :: expct
  expct(1, 1, 1) = 1.18973149535723176508575932662800702E+4932_16
  expct(2, 1, 1) = 0.999999999999999999999999999999999904_16
  expct(3, 1, 1) = 1.00000000000000000000000000000000019_16
  expct(3, 2, 1) = 256.000000000000000000000000000000000_16
  expct(3, 2, 2) = 3125.00000000000000000000000000000000_16
  expct(3, 2, 3) = 46656.0000000000000000000000000000000_16
end subroutine
