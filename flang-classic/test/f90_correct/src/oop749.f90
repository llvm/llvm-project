! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program p
  logical rslt(1),expect(1)
  integer, allocatable :: a(:)
  integer c(10)

  c = 99
  
  allocate(a(10), mold=c)

  rslt(1) = size(a,1) .eq. size(c,1)

  expect = .true.
  call check(rslt,expect,1)


  end
