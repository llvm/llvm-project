!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
program dt70
  implicit none
  integer :: i, j, k, actual(32), expected(32), curr = 0

  type dt1
    integer, allocatable :: m1
  end type

  type dt2
    type(dt1), allocatable :: m2(:)
  end type

  type(dt2), allocatable :: x1(:), x2(:,:)

  allocate(x1(1:4))
  allocate(x2(1:4, 1:4))

  do j = 1, 4
    allocate(x1(j)%m2(2:3))
    do k = 2, 3
      allocate(x1(j)%m2(k)%m1)
      x1(j)%m2(k)%m1 = 100*i + 10*j + k
    end do
  end do

  do i = 1, 4
    do j = 1, 4
      allocate(x2(i, j)%m2(2:3))
      do k = 2, 3
        allocate(x2(i, j)%m2(k)%m1)
        x2(i, j)%m2(k)%m1 = 100*i + 10*j + k
      end do
    end do
  end do

  call test_1_dim(0)
  call test_1_dim(10)
  call test_2_dim(0)
  ! problem case: 2 dimensional with 2nd dim of y different
  call test_2_dim(10)
  call check(actual, expected, curr)

contains

  subroutine test_1_dim(offset)
    integer, intent(in) :: offset
    type(dt2), allocatable :: y(:)
    allocate(y(offset+1:offset+4))
    y(:) = x1(:)
    do j = 1, 4
      do k = 2, 3
        curr = curr + 1
        actual(curr) = y(offset+j)%m2(k)%m1
        expected(curr) = 10*j + k
      end do
    end do
  end subroutine

  subroutine test_2_dim(offset)
    integer, intent(in) :: offset
    type(dt2), allocatable :: y(:,:)
    allocate(y(1:4, offset+1:offset+4))
    y(1,:) = x2(1,:)
    do j = 1, 4
      do k = 2, 3
        curr = curr + 1
        actual(curr) = y(1, offset+j)%m2(k)%m1
        expected(curr) = 100*1 + 10*j + k
      end do
    end do
  end subroutine

end program
