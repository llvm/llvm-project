!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program implied_do10
  implicit none
  real :: x(2) = 0
  integer :: i

  x = (/ (f(1), i = 1, 2) /)
  print *, x
contains
  function f(n)
    implicit none
    integer :: n
    real f(n)

    f = x(1) + 4
  end function
end
