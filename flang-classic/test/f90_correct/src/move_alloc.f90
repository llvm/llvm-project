!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program move_alloc_test
  implicit none
  type some
    integer, allocatable :: a(:)
  end type
  type some_more
    type(some) :: b
  end type
  type some_more_more
    type(some_more) :: c
  end type
  integer, parameter :: num = 7
  integer :: results(num)
  integer, parameter :: expect(num) = &
    (/.false., .false., .false., .false., .false., .true., .false./)
  integer, allocatable :: a(:)
  integer, allocatable :: b(:)
  type(some) :: c
  type(some_more) :: d
  type(some_more_more) :: e

  allocate(a(128))
  call move_alloc(a, b)
  deallocate(b)
  allocate(c%a(128))
  call move_alloc(c%a, b)
  deallocate(b)
  allocate(d%b%a(128))
  call move_alloc(d%b%a, b)
  deallocate(b)
  allocate(e%c%b%a(128))
  call move_alloc(e%c%b%a, b)
  deallocate(b)
  allocate(a(128))
  call sub0(a)
  allocate(c%a(128))
  call sub1(c)
  allocate(d%b%a(128))
  call sub2(d)
  results(1) = allocated(a)
  results(2) = allocated(b)
  results(3) = allocated(c%a)
  results(4) = allocated(d%b%a)
  results(5) = allocated(e%c%b%a)
  if (all( expect .eq. results)) then
    print *, "expect vs results match"
  else
    print *, "expect vs results mismatch"
  endif
  call check(results, expect, num)

contains

  subroutine sub0(a)
    implicit none
    integer, allocatable, intent(inout) :: a(:)
    integer, allocatable :: b(:)

    call move_alloc(a, b)
    deallocate(b)
  end subroutine

  subroutine sub1(par)
    implicit none
    type(some), intent(inout) :: par
    integer, allocatable :: b(:)

    call move_alloc(par%a, b)
    deallocate(b)
  end subroutine

  subroutine sub2(par)
    implicit none
    type(some_more), intent(inout) :: par
    integer, allocatable :: b(:)

    results(6) = allocated(par%b%a)
    call move_alloc(par%b%a, b)
    results(7) = allocated(par%b%a)
    deallocate(b)
  end subroutine

end program
