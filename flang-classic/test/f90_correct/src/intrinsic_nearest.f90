!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This test case is test for NEAREST intrinsic function(float and double precision).

program test_nearest
  real(kind=4) :: res4, infr4, maxr4
  real(kind=8) :: res8, infr8, maxr8
  integer(kind=4) :: infi4, maxi4
  integer(kind=8) :: infi8, maxi8
  equivalence (infr4, infi4)
  equivalence (maxr4, maxi4)
  equivalence (infr8, infi8)
  equivalence (maxr8, maxi8)

  infi4 = int(z'7f800000')
  maxi4 = int(z'7f7fffff')
  infi8 = int(z'7ff0000000000000', kind = 8)
  maxi8 = int(z'7fefffffffffffff', kind = 8)

  res4 = nearest(-infr4, 1.0)
  if (res4 .ne. -maxr4) STOP 1
  res4 = nearest(infr4, -1.0)
  if (res4 .ne. maxr4) STOP 2
  res8 = nearest(-infr8, 1.0_8)
  if (res8 .ne. -maxr8) STOP 3
  res8 = nearest(infr8, -1.0_8)
  if (res8 .ne. maxr8) STOP 4

  write(*,*) 'PASS'
end program
