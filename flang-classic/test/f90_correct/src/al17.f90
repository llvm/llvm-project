!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Sourced allocate of derived type with nested derived type.
program P
  type A
    integer :: a1
    integer, allocatable :: a2
    integer, allocatable :: a3(:)
  end type
  type B
    integer :: b1
    integer, allocatable :: b2
    type(A) :: b3
    type(A), allocatable :: b4
  end type
  type(B), allocatable :: x, y

  allocate(x)
  allocate(x%b4)
  allocate(x%b3%a3(3))
  x%b1 = 1
  x%b2 = 2
  x%b3%a1 = 3
  x%b3%a2 = 4
  x%b3%a3(:) = 9
  x%b4%a1 = 5
  x%b4%a2 = 6

  allocate(y, source=x)

  x%b1 = -1
  x%b2 = -2
  x%b3%a1 = -3
  x%b3%a2 = -4
  x%b3%a3(:) = -9
  x%b4%a1 = -5
  x%b4%a2 = -6
  call check( &
    [y%b1, y%b2, y%b3%a1, y%b3%a2, y%b4%a1, y%b4%a2, y%b3%a3(:)], &
    [1, 2, 3, 4, 5, 6, 9, 9, 9], 9)
end program
