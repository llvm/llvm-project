!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program dt68
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

  type(t3), allocatable :: z

  z = test()

  call check((/ z%i3, z%x3%i2, z%x3%x2%i1, z%x3%x2%x1 /), &
             (/ 1,    2,       3,          4          /), 4)

contains

  function test() result(z)
    type(t1) :: x
    type(t3), allocatable :: z
    x%i1 = 3
    allocate(x%x1, source=4)
    z = t3(1, t2(2, x))
  end function

end program
