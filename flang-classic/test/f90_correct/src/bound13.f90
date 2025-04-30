!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND of assumed-shape formal in type specification.

program test
  implicit none
  integer :: x(2,3,4)
  integer, allocatable :: y(:)
  type t
    integer, allocatable :: arr(:)
  end type
  type(t) :: z(8)
  integer :: i

  y = test_binary_expr(3, 4, 5, x)
  if (size(y) /= 910 .or. any(y /= 1)) STOP 1
  y = test_unary_expr(1, 2, 3, x)
  if (size(y) /= 288 .or. any(y /= 2)) STOP 2

  do i = 1, 8
    z(i)%arr = i * [1, 2, 3, 4, 5, 6, 7, 8, 9]
  enddo
  y = test_subscript_expr(1, 2, -2, x, z)
  if (size(y) /= 8 .or. any(y /= 3)) STOP 3
  print *, "PASS"
contains
  function test_binary_expr(l1, l2, l3, a) result(res)
    integer :: l1, l2, l3
    integer :: a(l1:, l2:, l3:)
    integer :: res(1:product(ubound(a) + lbound(a)))
    res = 1
  end function

  function test_unary_expr(l1, l2, l3, a) result(res)
    integer :: l1, l2, l3
    integer :: a(l1:, l2:, l3:)
    integer :: res(1:product(-ubound(a)) * sum(-lbound(a)))
    res = 2
  end function

  function test_subscript_expr(l1, l2, l3, a, b) result(res)
    integer :: l1, l2, l3
    integer :: a(l1:, l2:, l3:)
    type(t) :: b(:)
    integer :: res(1: b(product(ubound(a)))%arr(sum(lbound(a))))
    res = 3
  end function
end program
