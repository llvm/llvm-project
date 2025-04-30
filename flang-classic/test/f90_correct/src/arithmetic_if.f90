!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! tests arithmetic if codegen

function aif(x, expect)
  integer :: x
  integer :: expect(3)
  integer :: res
  if(x) 5,15,25
5   res = expect(1)
    goto 35
15  res = expect(2)
    goto 35
25  res = expect(3)
35  aif = res
end function

program main
  integer, parameter :: val1=10
  integer, parameter :: val2=20
  integer, parameter :: val3=30
  integer :: res(3)
  integer :: expect(3) = (/10,20,30/)
  res(1) = aif(-1, expect)
  res(2) = aif(0, expect)
  res(3) = aif(1, expect)
  call check(res, expect, 3)
end program
