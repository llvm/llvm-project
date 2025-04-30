! Part of the LLVM Project, under the Apache License v2.0
! See https://llvm.org/LICENSE.txt for license informatio
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test RADIX intrinsic with quad-precision arguments

program main
  integer, parameter :: eres(25) = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  integer :: res(25)
  integer :: i

  integer :: A(5) = [radix(-1.17E4932_16), &
  radix(-3.362E-4932_16), radix(0.0_16), &
  radix(3.362E-4932_16), radix(1.17E4932_16)]

  integer , parameter :: B(5) = [radix(-1.17E4932_16), &
  radix(-3.362E-4932_16), radix(0.0_16), &
  radix(3.362E-4932_16), radix(1.17E4932_16)]

  real(16) :: c1 = -1.17E4932_16, c2 = -3.362E-4932_16, c3 = 0.0_16, &
  c4 = 3.362E-4932_16, c5 = 1.17E4932_16
  integer :: C(5) = [radix(c1), radix(c2), radix(c3), radix(c4), radix(c5)]

  real(16), parameter :: e1 = -1.17E4932_16, e2 = -3.362E-4932_16, e3 = 0.0_16, &
  e4 = 3.362E-4932_16, e5 = 1.17E4932_16
  integer :: E(5) = [radix(e1), radix(e2), radix(e3), radix(e4), radix(e5)]

  real(16) :: D1(5) = [-1.17E4932_16, -3.362E-4932_16, 0.0_16, 3.362E-4932_16, 1.17E4932_16]
  integer :: D(5) = radix(D1)

  do i = 1,25
  if (i <= 5) then
    res(i) = A(i)
  else if (i <= 10) then
    res(i) = B(i-5)
  else if (i <= 15) then
    res(i) = C(i-10)
  else if (i <= 20) then
    res(i) = D(i-15)
  else
    res(i) = E(i-20)
  end if
  end do
call check(res, eres, 25)
end program main
