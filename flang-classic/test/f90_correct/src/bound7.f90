!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND when the array is assumed-shape.
! The options -Hy,68,1 and -My,68,1 are required.

program test
  implicit none
  integer, allocatable :: x(:, :, :, :)
  integer, parameter :: default_kind = kind(x)

  allocate(x(1:2, 2:4, 3:6, 4:8))
  call test_assumed_shape(x)
  call test_assumed_shape_with_lower_bound(1, 3, 2, 4, x)
  print *, "PASS"
contains
  subroutine test_assumed_shape(a)
    integer :: a(:, :, :, :)
    integer :: l_exp(4), u_exp(4), i

    l_exp = [1, 1, 1, 1]
    u_exp = [2, 3, 4, 5]
    ! DIM is constant
    if (any([lbound(a, 1), lbound(a, 2), lbound(a, 3), lbound(a, 4)] /= &
        l_exp)) STOP 1
    if (any([ubound(a, 1), ubound(a, 2), ubound(a, 3), ubound(a, 4)] /= &
        u_exp)) STOP 2

    ! DIM is variable
    do i = 1, 4
      if (kind(lbound(a, i)) /= default_kind .or. lbound(a, i) /= &
          l_exp(i)) STOP 3
      if (kind(lbound(a, i, kind=1)) /= 1 .or. lbound(a, i, kind=1) /= &
          l_exp(i)) STOP 4
      if (kind(lbound(a, i, kind=2)) /= 2 .or. lbound(a, i, kind=2) /= &
          l_exp(i)) STOP 5
      if (kind(lbound(a, i, kind=4)) /= 4 .or. lbound(a, i, kind=4) /= &
          l_exp(i)) STOP 6
      if (kind(lbound(a, i, kind=8)) /= 8 .or. lbound(a, i, kind=8) /= &
          l_exp(i)) STOP 7

      if (kind(ubound(a, i)) /= default_kind .or. ubound(a, i) /= &
          u_exp(i)) STOP 9
      if (kind(ubound(a, i, kind=1)) /= 1 .or. ubound(a, i, kind=1) /= &
          u_exp(i)) STOP 10
      if (kind(ubound(a, i, kind=2)) /= 2 .or. ubound(a, i, kind=2) /= &
          u_exp(i)) STOP 11
      if (kind(ubound(a, i, kind=4)) /= 4 .or. ubound(a, i, kind=4) /= &
          u_exp(i)) STOP 12
      if (kind(ubound(a, i, kind=8)) /= 8 .or. ubound(a, i, kind=8) /= &
          u_exp(i)) STOP 13
    enddo

    ! DIM is missing
    if (kind(lbound(a)) /= default_kind .or. size(lbound(a)) /= 4 .or. &
        any(lbound(a) /= l_exp)) STOP 15
    if (kind(lbound(a, kind=1)) /= 1 .or. size(lbound(a, kind=1)) /= 4 .or. &
        any(lbound(a, kind=1) /= l_exp)) STOP 16
    if (kind(lbound(a, kind=2)) /= 2 .or. size(lbound(a, kind=2)) /= 4 .or. &
        any(lbound(a, kind=2) /= l_exp)) STOP 17
    if (kind(lbound(a, kind=4)) /= 4 .or. size(lbound(a, kind=4)) /= 4 .or. &
        any(lbound(a, kind=4) /= l_exp)) STOP 18
    if (kind(lbound(a, kind=8)) /= 8 .or. size(lbound(a, kind=8)) /= 4 .or. &
        any(lbound(a, kind=8) /= l_exp)) STOP 19

    if (kind(ubound(a)) /= default_kind .or. size(ubound(a)) /= 4 .or. &
        any(ubound(a) /= u_exp)) STOP 21
    if (kind(ubound(a, kind=1)) /= 1 .or. size(ubound(a, kind=1)) /= 4 .or. &
        any(ubound(a, kind=1) /= u_exp)) STOP 22
    if (kind(ubound(a, kind=2)) /= 2 .or. size(ubound(a, kind=2)) /= 4 .or. &
        any(ubound(a, kind=2) /= u_exp)) STOP 23
    if (kind(ubound(a, kind=4)) /= 4 .or. size(ubound(a, kind=4)) /= 4 .or. &
        any(ubound(a, kind=4) /= u_exp)) STOP 24
    if (kind(ubound(a, kind=8)) /= 8 .or. size(ubound(a, kind=8)) /= 4 .or. &
        any(ubound(a, kind=8) /= u_exp)) STOP 25
  end subroutine test_assumed_shape

  subroutine test_assumed_shape_with_lower_bound(l1, l2, l3, l4, a)
    integer :: l1, l2, l3, l4
    integer :: a(l1:, l2:, l3:, l4:)
    integer :: l_exp(4), u_exp(4), i

    l_exp = [l1, l2, l3, l4]
    u_exp = [l1 + 1, l2 + 2, l3 + 3, l4 + 4]
    ! DIM is constant
    if (any([lbound(a, 1), lbound(a, 2), lbound(a, 3), lbound(a, 4)] /= &
        l_exp)) STOP 27
    if (any([ubound(a, 1), ubound(a, 2), ubound(a, 3), ubound(a, 4)] /= &
        u_exp)) STOP 28

    ! DIM is variable
    do i = 1, 4
      if (kind(lbound(a, i)) /= default_kind .or. lbound(a, i) /= &
          l_exp(i)) STOP 29
      if (kind(lbound(a, i, kind=1)) /= 1 .or. lbound(a, i, kind=1) /= &
          l_exp(i)) STOP 30
      if (kind(lbound(a, i, kind=2)) /= 2 .or. lbound(a, i, kind=2) /= &
          l_exp(i)) STOP 31
      if (kind(lbound(a, i, kind=4)) /= 4 .or. lbound(a, i, kind=4) /= &
          l_exp(i)) STOP 32
      if (kind(lbound(a, i, kind=8)) /= 8 .or. lbound(a, i, kind=8) /= &
          l_exp(i)) STOP 33

      if (kind(ubound(a, i)) /= default_kind .or. ubound(a, i) /= &
          u_exp(i)) STOP 35
      if (kind(ubound(a, i, kind=1)) /= 1 .or. ubound(a, i, kind=1) /= &
          u_exp(i)) STOP 36
      if (kind(ubound(a, i, kind=2)) /= 2 .or. ubound(a, i, kind=2) /= &
          u_exp(i)) STOP 37
      if (kind(ubound(a, i, kind=4)) /= 4 .or. ubound(a, i, kind=4) /= &
          u_exp(i)) STOP 38
      if (kind(ubound(a, i, kind=8)) /= 8 .or. ubound(a, i, kind=8) /= &
          u_exp(i)) STOP 39
    enddo

    ! DIM is missing
    if (kind(lbound(a)) /= default_kind .or. size(lbound(a)) /= 4 .or. &
        any(lbound(a) /= l_exp)) STOP 41
    if (kind(lbound(a, kind=1)) /= 1 .or. size(lbound(a, kind=1)) /= 4 .or. &
        any(lbound(a, kind=1) /= l_exp)) STOP 42
    if (kind(lbound(a, kind=2)) /= 2 .or. size(lbound(a, kind=2)) /= 4 .or. &
        any(lbound(a, kind=2) /= l_exp)) STOP 43
    if (kind(lbound(a, kind=4)) /= 4 .or. size(lbound(a, kind=4)) /= 4 .or. &
        any(lbound(a, kind=4) /= l_exp)) STOP 44
    if (kind(lbound(a, kind=8)) /= 8 .or. size(lbound(a, kind=8)) /= 4 .or. &
        any(lbound(a, kind=8) /= l_exp)) STOP 45

    if (kind(ubound(a)) /= default_kind .or. size(ubound(a)) /= 4 .or. &
        any(ubound(a) /= u_exp)) STOP 47
    if (kind(ubound(a, kind=1)) /= 1 .or. size(ubound(a, kind=1)) /= 4 .or. &
        any(ubound(a, kind=1) /= u_exp)) STOP 48
    if (kind(ubound(a, kind=2)) /= 2 .or. size(ubound(a, kind=2)) /= 4 .or. &
        any(ubound(a, kind=2) /= u_exp)) STOP 49
    if (kind(ubound(a, kind=4)) /= 4 .or. size(ubound(a, kind=4)) /= 4 .or. &
        any(ubound(a, kind=4) /= u_exp)) STOP 50
    if (kind(ubound(a, kind=8)) /= 8 .or. size(ubound(a, kind=8)) /= 4 .or. &
        any(ubound(a, kind=8) /= u_exp)) STOP 51
  end subroutine test_assumed_shape_with_lower_bound
end program
