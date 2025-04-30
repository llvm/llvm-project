!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND when the array is deferred-shape.
! The options -Hy,68,1 and -My,68,1 are required.

program test
  implicit none
  integer, allocatable :: x(:, :, :, :)
  integer, pointer :: y(:, :, :, :)
  integer :: i
  integer :: n
  integer, parameter :: default_kind = kind(i)

  n = 2
  x = reshape([(i, i=1,120)], [2, 3, 4, 5])
  allocate(y(1:2, 2:4, 3:6, 4:8))

  call test_dim_present(x, y)
  call test_dim_missing(x, y)
  call test_component()
  print *, "PASS"
contains
  subroutine test_dim_present(a, b)
    integer, allocatable :: a(:, :, :, :)
    integer, pointer :: b(:, :, :, :)
    if (kind(ubound(a, 1)) /= default_kind .or. ubound(a, 1) /= 2) STOP 1
    if (kind(ubound(a, 1, 1)) /= 1 .or. ubound(a, 1, 1) /= 2) STOP 2
    if (kind(ubound(a, n, 2)) /= 2 .or. ubound(a, n, 2) /= 3) STOP 3
    if (kind(ubound(a, n+1, 4)) /= 4 .or. ubound(a, n+1, 4) /= 4) STOP 4
    if (kind(ubound(a, 4, 8)) /= 8 .or. ubound(a, 4, 8) /= 5) STOP 5
    if (kind(lbound(b, 1)) /= default_kind .or. lbound(b, 1) /= 1) STOP 6
    if (kind(lbound(b, 1, 1)) /= 1 .or. lbound(b, 1, 1) /= 1) STOP 7
    if (kind(lbound(b, n, 2)) /= 2 .or. lbound(b, n, 2) /= 2) STOP 8
    if (kind(lbound(b, n+1, 4)) /= 4 .or. lbound(b, n+1, 4) /= 3) STOP 9
    if (kind(lbound(b, 4, 8)) /= 8 .or. lbound(b, 4, 8) /= 4) STOP 10
  end subroutine

  subroutine test_dim_missing (a, b)
    integer, allocatable :: a(:, :, :, :)
    integer, pointer :: b(:, :, :, :)
    if (kind(ubound(a)) /= default_kind .or. size(ubound(a)) /= 4 .or. &
        any(ubound(a) /= [2, 3, 4, 5])) STOP 11
    if (kind(ubound(a, kind=1)) /= 1 .or. size(ubound(a, kind=1)) /= 4 .or. &
        any(ubound(a, kind=1) /= [2, 3, 4, 5])) STOP 12
    if (kind(ubound(a, kind=2)) /= 2 .or. size(ubound(a, kind=2)) /= 4 .or. &
        any(ubound(a, kind=2) /= [2, 3, 4, 5])) STOP 13
    if (kind(ubound(a, kind=4)) /= 4 .or. size(ubound(a, kind=4)) /= 4 .or. &
        any(ubound(a, kind=4) /= [2, 3, 4, 5])) STOP 14
    if (kind(ubound(a, kind=8)) /= 8 .or. size(ubound(a, kind=8)) /= 4 .or. &
        any(ubound(a, kind=8) /= [2, 3, 4, 5])) STOP 15
    if (kind(lbound(b)) /= default_kind .or. size(lbound(b)) /= 4 .or. &
        any(lbound(b) /= [1, 2, 3, 4])) STOP 16
    if (kind(lbound(b, kind=1)) /= 1 .or. size(lbound(b, kind=1)) /= 4 .or. &
        any(lbound(b, kind=1) /= [1, 2, 3, 4])) STOP 17
    if (kind(lbound(b, kind=2)) /= 2 .or. size(lbound(b, kind=2)) /= 4 .or. &
        any(lbound(b, kind=2) /= [1, 2, 3, 4])) STOP 18
    if (kind(lbound(b, kind=4)) /= 4 .or. size(lbound(b, kind=4)) /= 4 .or. &
        any(lbound(b, kind=4) /= [1, 2, 3, 4])) STOP 19
    if (kind(lbound(b, kind=8)) /= 8 .or. size(lbound(b, kind=8)) /= 4 .or. &
        any(lbound(b, kind=8) /= [1, 2, 3, 4])) STOP 20
  end subroutine

  subroutine test_component()
    type t
      integer, allocatable :: x(:, :, :, :)
      integer, pointer :: y(:, :, :, :)
    end type
    type(t) :: a
    a%x = reshape([(i, i=1,120)], [2, 3, 4, 5])
    allocate(a%y(1:2, 2:4, 3:6, 4:8))
    if (any(lbound(a%x) /= 1)) STOP 21
    if (any(ubound(a%x) /= [2, 3, 4, 5])) STOP 22
    if (any(lbound(a%y) /= [1, 2, 3, 4])) STOP 23
    if (any(ubound(a%y) /= [2, 4, 6, 8])) STOP 24
  end subroutine
end program
