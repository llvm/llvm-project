!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND of assumed-shape formal in type specification when
! DIM is present.

program test
  implicit none
  integer :: x(2,3,4)
  integer, allocatable :: x_alloc(:, :, :)
  integer, pointer :: x_ptr(:, :, :)
  integer, allocatable :: arr_res(:)
  character(len=:), allocatable :: char_res
  integer :: i

  allocate(x_alloc(2:3, 3:5, 4:7))
  allocate(x_ptr(2:3, 3:5, 4:7))

  arr_res = array_test_specified_lb(1, 0, -1, x)
  if (size(arr_res) /= 8 .or. any(arr_res /= 1)) STOP 1
  arr_res = array_test_specified_lb(2, 1, 0, x(2:, 3:, 2:))
  if (size(arr_res) /= 4 .or. any(arr_res /= 1)) STOP 2
  arr_res = array_test_missing_lb(x_alloc)
  if (size(arr_res) /= 3 .or. any(arr_res /= 2)) STOP 3
  arr_res = array_test_specified_lb(1, 0, -1, x_alloc)
  if (size(arr_res) /= 8 .or. any(arr_res /= 1)) STOP 4
  arr_res = array_test_missing_lb(x_ptr)
  if (size(arr_res) /= 3 .or. any(arr_res /= 2)) STOP 5

  char_res = char_test_specified_lb(1, 0, -1, x)
  if (len(char_res) /= 8 .or. char_res /= 'aaaaaaaa') STOP 6
  char_res = char_test_specified_lb(2, 1, 0, x(2:, 3:, 2:))
  if (len(char_res) /= 4 .or. char_res /= 'aaaa') STOP 7
  char_res = char_test_missing_lb(x_alloc)
  if (len(char_res) /= 3 .or. char_res /= 'bbb') STOP 8
  char_res = char_test_specified_lb(1, 0, -1, x_alloc)
  if (len(char_res) /= 8 .or. char_res /= 'aaaaaaaa') STOP 9
  char_res = char_test_missing_lb(x_ptr)
  if (len(char_res) /= 3 .or. char_res /= 'bbb') STOP 10

  arr_res = test_noncnst_dim(1, 0, -1, x, 1, 2, 3)
  if (size(arr_res) /= 8 .or. any(arr_res /= 1)) STOP 11

  print *, "PASS"
contains
  function array_test_specified_lb(l1, l2, l3, a) result(res)
    integer :: l1, l2, l3
    integer :: a(l1:, l2:, l3:)
    integer :: res(1:ubound(a, 1) * ubound(a, 2) * ubound(a, 3))
    res = 1
  end function

  function array_test_missing_lb(a) result(res)
    integer :: a(:, :, :)
    integer :: res(1:lbound(a, 1) + lbound(a, 2) + lbound(a, 3))
    res = 2
  end function

  function char_test_specified_lb(l1, l2, l3, a) result(res)
    integer :: l1, l2, l3
    integer :: a(l1:, l2:, l3:)
    character(len=ubound(a, 1) * ubound(a, 2) * ubound(a, 3)) :: res
    res = repeat('a', ubound(a, 1) * ubound(a, 2) * ubound(a, 3))
  end function

  function char_test_missing_lb(a) result(res)
    integer :: a(:, :, :)
    character(len=lbound(a, 1) + lbound(a, 2) + lbound(a, 3)) :: res
    res = repeat('b', lbound(a, 1) + lbound(a, 2) + lbound(a, 3))
  end function

  function test_noncnst_dim(l1, l2, l3, a, d1, d2, d3) result(res)
    integer :: l1, l2, l3
    integer :: a(l1:, l2:, l3:)
    integer :: d1, d2, d3
    integer :: res(1:ubound(a, d1) * ubound(a, d2) * ubound(a, d3))
    res = 1
  end function
end program
