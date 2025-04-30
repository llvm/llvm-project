!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test allocatable assignment.
! Requires -Mallocatable=03
program dt69
  implicit none
  integer, parameter :: N = 10
  logical :: res(N), exp(N) = .true.
  integer :: curr = 0

  type dt1
    integer, allocatable :: m1
  end type
  type dt2
    type(dt1), allocatable :: m2(:)
  end type

  !call test1()
  call test2()
  call test3()
  call test4()
  call check(res, exp, curr)

contains

  !subroutine test1()
  !  integer*8 :: prev_loc
  !  type(dt1), allocatable :: x(:), y(:)
  !  allocate(x(100))
  !  allocate(y(1))
  !  prev_loc = loc(y)
  !  ! y is allocated but not conformable so must be reallocated
  !  y = x
  !  curr = curr + 1
  !  !NOTE: this test could fail if the newly allocated y gets the same memory
  !  !location as the old one
  !  res(curr) = prev_loc .ne. loc(y)
  !end subroutine

  subroutine test2()
    integer*8 :: prev_loc
    type(dt1), allocatable :: x(:), y(:)
    allocate(x(2))
    allocate(y(2))
    prev_loc = loc(y)
    ! y is allocated and conformable so need not be reallocated
    y = x
    curr = curr + 1
    res(curr) = prev_loc .eq. loc(y)
  end subroutine

  ! Like test2 but source is not allocatable
  subroutine test3()
    integer*8 :: prev_loc
    type(dt1) :: x(2)
    type(dt1), allocatable :: y(:)
    allocate(y(2))
    prev_loc = loc(y)
    ! y is allocated and conformable so need not be reallocated
    y = x
    curr = curr + 1
    res(curr) = prev_loc .eq. loc(y)
  end subroutine

  subroutine test4()
    type(dt2), allocatable :: x(:), y(:)
    allocate(x(1))
    allocate(y(1))
    ! y(1)%m2 should not be allocated after this assignment
    y(1)%m2 = x(1)%m2
    curr = curr + 1
    res(curr) = .not. allocated(y(1)%m2)
  end subroutine

end program
