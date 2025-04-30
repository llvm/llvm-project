! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check IF statement with real*16

PROGRAM test
IMPLICIT NONE
  real(16) :: x
  integer :: result(2), expect(2)
  expect = 1
  result = 0
  x = -1.0

  if(x .lt. 0.0) goto 100
  100 result(1) = 1

  if(x) 10,20,30
  10 result(2) = 0
  20 result(2) = 0
  30 result(2) = 1

  call check(result, expect, 2)

end PROGRAM test
