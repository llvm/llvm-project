!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for empty type.

module m
implicit none
  type empty
  end type empty
contains
  integer function func(a)
    type(empty), intent(in) :: a(:)
    character(10) :: str = 'abcdefghij'
    write(str, *) a
    if (str /= '') STOP 1
    func = size(a)
  end
end module m

program test
    use m
    implicit none
    integer, parameter :: n = 3
    integer :: rst(n), expect(n)
    type(empty), parameter :: e1 = empty()
    type(empty), parameter :: e3(3) = [empty(), empty(), e1]

    rst = 0
    expect(1) = 1
    expect(2) = 2
    expect(3) = 3

    rst(1) = func([empty()])
    rst(2) = func([empty(), e1])
    rst(3) = func(e3)

    call check(rst, expect, n)
end program test
