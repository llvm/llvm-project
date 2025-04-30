! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module mod
type, abstract :: base
  integer :: z = 101
end type
type, extends(base) :: t
  integer :: i
end type
end module

program p
  use mod
  logical rslt(3),expect(3)
  class(base), allocatable :: a(:)
  class(t), allocatable :: b(:)
  type(t) :: o(10)

  allocate(a, b, mold=o)

  select type(p => b)
  type is (t)
  print *, p 
  rslt(1) = .true.
  class default
  rslt(1) = .false.
  end select

  print *

  select type(p => b)
  type is (t)
  print *, p 
  rslt(2) = .true.
  class default
  rslt(2) = .false.
  end select

  rslt(3) = same_type_as(a,b)

  expect = .true.
  call check(rslt,expect,3)

  deallocate(a,b)
  end
