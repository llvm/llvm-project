!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Sourced allocate of array of derived types.
program P
  implicit none
  integer, parameter :: SIZE1 = 5, SIZE2 = 3
  type A
    integer :: a
  end type A
  type B
    integer, allocatable :: bb
    type(A), allocatable :: b
  end type B

  call test_1_dim()
  call test_2_dim()

contains

  subroutine test_1_dim()
    integer :: i, n
    integer :: expected(SIZE1), actual(SIZE1)
    type(B), allocatable :: x(:), y(:)
    allocate(x(SIZE1))
    do i = 1, SIZE1
      allocate(x(i)%b)
      x(i)%b%a = i
    end do

    allocate(y, source=x)

    n = 0
    do i = 1, SIZE1
      n = n + 1
      expected(n) = x(i)%b%a
      x(i)%b%a = 0  ! should not change y
      actual(n) = y(i)%b%a
    end do
    call check(actual, expected, SIZE1)
  end subroutine

  subroutine test_2_dim()
    integer :: i, j, n
    integer :: expected(SIZE1*SIZE2), actual(SIZE1*SIZE2)
    type(B), allocatable :: x(:,:), y(:,:)

    allocate(x(SIZE1,SIZE2))
    do i = 1, SIZE1
      do j = 1, SIZE2
        allocate(x(i,j)%b)
        x(i,j)%b%a = 10*i + j
      end do
    end do

    allocate(y, source=x)

    n = 0
    do i = 1, SIZE1
      do j = 1, SIZE2
        n = n + 1
        expected(n) = x(i,j)%b%a
        x(i,j)%b%a = 0  ! should not change y
        actual(n) = y(i,j)%b%a
      end do
    end do
    call check(actual, expected, SIZE1*SIZE2)
  end subroutine

end program
