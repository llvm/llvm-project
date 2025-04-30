! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Similar to oop749 but has multiple mold= allocations.

program p
  logical rslt(2),expect(2)
  integer, allocatable :: a(:), b(:)
  integer c(10)

  c = 99
  
  allocate(a(10), b, mold=c)

  rslt(1) = size(b,1) .eq. size(c,1)
  rslt(2) = size(a,1) .eq. size(b,1)

  expect = .true.
  call check(rslt,expect,2)


  end
