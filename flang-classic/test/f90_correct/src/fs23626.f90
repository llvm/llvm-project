! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
program a
  implicit none

  type :: t1
     integer, dimension(:), allocatable :: mem1
  end type
  type :: t2
     type(t1), dimension(:), allocatable :: mem2
  end type

  integer, dimension(2) :: i = (/2, 3/)
  type(t2) :: x
  type(t2) :: y
  allocate(x%mem2(8))

  x%mem2(1)%mem1 = (/11, 12, 13, 14/)
  x%mem2(2)%mem1 = (/21, 22, 23, 24/)
  x%mem2(3)%mem1 = (/31, 32, 33, 34/)

  y%mem2 = x%mem2(i)
  if (y%mem2(1)%mem1(1) /= 21) stop "FAIL"
  stop "PASS"

end program
