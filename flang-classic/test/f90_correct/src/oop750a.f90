! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Same as oop750 but has multiple mold= allocations.

program p
  logical rslt(2),expect(2)
  integer, allocatable :: a(:), b(:)
  integer :: c(10)

  c = 101
  allocate(a(10), b(21:30), mold=101)

  rslt(1) = size(a,1) .eq. size(c,1) 
  rslt(2) = size(b,1) .eq. size(c,1) 

  expect = .true.
  call check(rslt,expect,2)


  end
