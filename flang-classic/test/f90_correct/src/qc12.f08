! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for exponentiation operation of quad complex : complex(16) ^ complex(16)

program test
  integer, parameter :: n = 6
  integer, parameter :: m = 2 * n
  real(16), parameter :: q_tol = 5E-33_16
  integer :: i
  complex(16) :: c1, c2
  complex(16) :: czero, cone, ctwo, chalf
  complex(16) :: rst(n)

  real(16) :: result(2*n), expect(2*n)

  c1 = (2.588797885226587678678452432135782132_16, -1.3387123548951275562114863159753156_16)
  c2 = (-4.90972372139829213745307352991854293_16, 6.93131142657099727375757869256872649_16)
  czero = (+0.0_16, -0.0_16)
  cone = (1.0_16, 1.0_16)
  ctwo = (2.0_16, 2.0_16)
  chalf = (0.5_16, 0.5_16)

  rst(1) = c1 ** c2
  rst(2) = c2 ** c1
  rst(3) = c1 ** czero
  rst(4) = c1 ** cone
  rst(5) = c1 ** ctwo
  rst(6) = c1 ** chalf

  expect(1) = -0.135297907708383681020369580959022084_16
  expect(2) = -4.674708490835295048296972011517677046E-0002_16
  expect(3) = -4473.97252807570383096961993695466412_16
  expect(4) = 1600.90929403054937750279687099358700_16
  expect(5) = 1.0_16
  expect(6) = -0.0_16
  expect(7) = 3.89657595790591187843508282003134820_16
  expect(8) = 2.62273276106394325211131133866734712_16
  expect(9) = 8.30457705977227949384385126248779368_16
  expect(10) = 20.4393548415479035559959866509411226_16
  expect(11) = 2.07287235801682060528062674866912322_16
  expect(12) = 0.632632479978938657264986278475722331_16

  do i = 1, m, 2
    result(i) = rst((i+1)/2)%re
  enddo

  do i = 2, m, 2
    result(i) = rst(i/2)%im
  enddo

  do i = 1, m
    if(expect(i) .eq. 0.0_16) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. q_tol) STOP i
    endif
  enddo

  print *, 'PASS'
end
