! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for IO take quad precision' intrinsic argument

program test
implicit none
  real(16) :: r16, a, b
  character(80) :: str1, str2
  integer :: result(2), expect(2)
  expect = 1
  result = 0
  r16 = 3.14156331_16

  write(str1,100) sin(1.23456789_16) 
  write(str2,100) tan(r16)
  if(str1 == "  0.944005725004534594548290920445127473E+000") result(1) = 1
  if(str2 == " -0.293435898016605254142090204181706412E-004") result(2) = 1
  100 FORMAT('',E)

  call check(result, expect, 2)

end program
