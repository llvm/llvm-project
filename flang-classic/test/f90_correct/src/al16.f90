!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Sourced allocate of derived type with nested allocatable array.
program P
  implicit none

  integer, parameter :: SIZE1 = 5, SIZE2 = 3
  type A
    integer, allocatable :: a
  end type
  type B1
    type(A), allocatable :: b(:)
  end type B1
  type B2
    type(A), allocatable :: b(:,:)
  end type B2
  type C1
    type(B1), allocatable :: c
  end type C1
  type C2
    type(B2), allocatable :: c
  end type C2

  call test_1_dim()
  call test_2_dim()

contains

  subroutine test_1_dim()
    integer :: i
    integer :: expected(SIZE1), actual(SIZE1)
    type(C1), allocatable :: x, y
    allocate(x)
    allocate(x%c)
    allocate(x%c%b(SIZE1))
    do i = 1, SIZE1
      x%c%b(i)%a = i
    end do
    expected = x%c%b(:)%a

    allocate(y, source=x)

    do i = 1, SIZE1
      x%c%b(i)%a = -i
    end do
    actual = y%c%b(:)%a
    call check(actual, expected, SIZE1)
  end subroutine

  subroutine test_2_dim()
    integer :: i, j, n
    integer :: expected(SIZE1*SIZE2), actual(SIZE1*SIZE2)
    type(C2), allocatable :: x, y
    allocate(x)
    allocate(x%c)
    allocate(x%c%b(SIZE1,SIZE2))
    do i = 1, SIZE1
      do j = 1, SIZE2
        x%c%b(i,j)%a = 10*i + j
      end do
    end do

    allocate(y, source=x)

    n = 0
    do i = 1, SIZE1
      do j = 1, SIZE2
        n = n + 1
        expected(n) = x%c%b(i,j)%a
        x%c%b(i,j)%a = 0
        actual(n) = y%c%b(i,j)%a
      end do
    end do
    call check(actual, expected, n)
  end subroutine

end program
