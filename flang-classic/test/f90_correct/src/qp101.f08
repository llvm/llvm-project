! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O of quad-precision values

program test

  type dtype1
    character :: c
    real(16), dimension(2, 2) :: ar16
  end type

  type dtype2
    character :: c
    complex(16), dimension(2, 2) :: ac16
  end type

  integer, parameter :: k = 16, n = 8
  real(16) :: rzero = 0.0_16
  real(16) :: rone = 1.0_16
  complex(16) :: czero = (0.0_16, 0.0_16)
  complex(16) :: cone = (1.0_16, 1.0_16)
  type(dtype1) :: r(n)
  type(dtype2) :: c(n)
  integer :: result(n), expect(n)
  expect = 1
  result = 0

  r(1)%ar16 = 0.0_16
  r(2)%ar16 = 1.0_16
  r(3)%ar16 = rzero
  r(4)%ar16 = rone
  c(1)%ac16 = (0.0_16, 0.0_16)
  c(2)%ac16 = (1.0_16, 1.0_16)
  c(3)%ac16 = czero
  c(4)%ac16 = cone
  do i = 1, 4
    print *, r(i)%ar16
    print *, c(i)%ac16
  enddo
  print *, 'PASS'
  !call checkr16(result, expect, n)

end program
