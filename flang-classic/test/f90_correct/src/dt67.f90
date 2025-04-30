!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program dt67
  implicit none

  type t1
    integer :: i1
    integer, allocatable :: x1
  end type t1

  type t2
    integer :: i2
    type(t1) :: x2
  end type t2

  type t3
    integer :: i3
    type(t2), allocatable :: x3
  end type t3

  call test()
  call check(1,1,1)

contains

  subroutine test()
    type(t1) :: z
    type(t3), allocatable :: y
    z%i1 = 1
    allocate(z%x1, source=2)
    y = t3(1, t2(1, z))
  end subroutine

end program
