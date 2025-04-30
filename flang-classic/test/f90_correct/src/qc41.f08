! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for operation of quad complex.

program test
  integer, parameter :: n = 12
  real(16) :: q_tol = 5e-33_16
  complex(16) :: result(n), expect(n)
  complex(16), parameter :: rst1(2) = (/(1.0_16, 2.0_16), (3.0_16, 4.0_16)/) + 2.0_16
  complex(16), parameter :: rst2(2) = (/(1.0_16, 2.0_16), (3.0_16, 4.0_16)/) - 2.0_16
  complex(16), parameter :: rst3(2) = (/(1.0_16, 2.0_16), (3.0_16, 4.0_16)/) * 2.0_16
  complex(16), parameter :: rst4(2) = (/(42.0_16, -7.0_16), (-1.0_16, 4.0_16)/) / (-1.0_16, 7.0_16)
  complex(16), parameter :: rst5(2) = (/(15.0_16, 9.0_16), (-1.0_16, 4.0_16)/) / (7.0_16, 12.0_16)
  complex(16), parameter :: rst6(2) = (/(1.0_16, 2.0_16), (1.0_16, 4.0_16)/) / (27.0_16, 13.0_16)
  logical, parameter :: rst7(2) = (/(1.0_16, 2.0_16), (-1.0_16, 2.0_16)/) == (1.0_16, 2.0_16)

  expect(1) = (3.0_16, 2.0_16)
  expect(2) = (5.0_16, 4.0_16)
  expect(3) = (-1.0_16, 2.0_16)
  expect(4) = (1.0_16, 4.0_16)
  expect(5) = (2.0_16, 4.0_16)
  expect(6) = (6.0_16, 8.0_16)
  expect(7) = (-1.82_16, -5.74_16)
  expect(8) = (0.579999999999999999999999999999999965_16,&
               5.999999999999999999999999999999999807E-0002_16)
  expect(9) = (1.10362694300518134715025906735751290_16,&
               -0.606217616580310880829015544041450762_16)
  expect(10) = (0.212435233160621761658031088082901549_16,&
                0.207253886010362694300518134715025913_16)
  expect(11) = (5.902004454342984409799554565701558953E-0002_16,&
                4.565701559020044543429844097995545934E-0002_16)
  expect(12) = (8.797327394209354120267260579064588464E-0002_16,&
                0.105790645879732739420935412026726054_16)

  result(1) = rst1(1)
  result(2) = rst1(2)
  result(3) = rst2(1)
  result(4) = rst2(2)
  result(5) = rst3(1)
  result(6) = rst3(2)
  result(7) = rst4(1)
  result(8) = rst4(2)
  result(9) = rst5(1)
  result(10) = rst5(2)
  result(11) = rst6(1)
  result(12) = rst6(2)

  do i = 1, n
    if (expect(i)%re .eq. 0.0_16) then
      if (result(i)%re .ne. expect(i)%re) STOP i
    else
      if (abs((result(i)%re - expect(i)%re) / expect(i)%re) .gt. q_tol) STOP i
    endif
    if (expect(i)%im .eq. 0.0_16) then
      if (result(i)%im .ne. expect(i)%im) STOP i
    else
      if (abs((result(i)%im - expect(i)%im) / expect(i)%im) .gt. q_tol) STOP i
    endif
  enddo
  if (rst7(1) .neqv. .true.) STOP n+1
  if (rst7(2) .neqv. .false.) STOP n+2

  print*, 'PASS'
end
