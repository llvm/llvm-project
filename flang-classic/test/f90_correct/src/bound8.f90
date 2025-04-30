!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND when the array is explicit-shape.

program test
  implicit none
  integer :: x(2,3,4)
  call test_explicit_shape(4, 3, 2, x)
  call test_explicit_shape_with_lower_bound(3, 5, 9, 12, 1, 5, x)
  call test_subobject()
  call test_array_func()
  print *, "PASS"
contains
  subroutine test_explicit_shape(n1, n2, n3, a)
    integer :: n1, n2, n3
    integer :: a(n1, n2, n3)
    if (any(lbound(a) /= 1)) STOP 1
    if (any(ubound(a) /= [n1, n2, n3])) STOP 2
  end subroutine

  subroutine test_explicit_shape_with_lower_bound(l1, u1, l2, u2, l3, u3, a)
    integer :: l1, u1, l2, u2, l3, u3
    integer :: a(l1:u1, l2:u2, l3:u3)
    if (any(lbound(a) /= [l1, l2, l3])) STOP 3
    if (any(ubound(a) /= [u1, u2, u3])) STOP 4
  end subroutine

  subroutine test_subobject()
    integer :: y(10, 10, 10)
    type t
      integer, allocatable :: y(:, :, :)
      integer :: i
    end type
    type(t) :: z, zz(2, 3, 4)
    if (any(lbound(y(3:4, 5:9, 1:5)) /= [1, 1, 1])) STOP 5
    if (any(ubound(y(3:4, 5:9, 1:5)) /= [2, 5, 5])) STOP 6

    allocate(z%y(10:20, 20:30, 30:40))
    if (any(lbound(z%y(13:14, 25:29, 31:35)) /= [1, 1, 1])) STOP 7
    if (any(ubound(z%y(13:14, 25:29, 31:35)) /= [2, 5, 5])) STOP 8

    if (any(lbound(zz%i) /= [1, 1, 1])) STOP 9
    if (any(ubound(zz%i) /= [2, 3, 4])) STOP 10
  end subroutine

  subroutine test_array_func()
    integer, allocatable :: y(:, :, :)
    allocate(y(-1:1, -2:2, -3:3))
    if (lbound(shape(y), 1) /= 1) STOP 11
    if (any(lbound(shape(y)) /= 1)) STOP 12
    if (ubound(shape(y), 1) /= 3) STOP 13
    if (any(ubound(shape(y)) /= 3)) STOP 14
  end subroutine
end program
