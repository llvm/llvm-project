! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module mod
type :: base(r)
  integer, kind :: r=12
  integer :: z
end type
type, extends(base) :: t
  integer :: i
end type
end module

program p
  use mod
  logical rslt(5),expect(5)
  type(base(20)), allocatable :: b(:)
  type(base(20)), allocatable :: z(:)
  type(base(20)), allocatable :: o(:)

  allocate(o(10))
  allocate(b, z, mold=o)
  
  rslt(1) = same_type_as(z,b)
  rslt(2) = size(b,1) .eq. size(o,1)
  rslt(3) = size(z,1) .eq. size(o,1)
 
  rslt(4) = .true.
  do i=1, size(b,1)
    if (b(i)%r .ne. o(i)%r) then
      rslt(3) = .false.
    endif
  enddo
  rslt(5) = .true.
  do i=1, size(z,1)
    if (z(i)%r .ne. o(i)%r) then
      rslt(5) = .false.
    endif
  enddo

  expect = .true.
  call check(rslt,expect,5)

  end
