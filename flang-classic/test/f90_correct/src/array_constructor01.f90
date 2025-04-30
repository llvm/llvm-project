!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for that the size of array section in array constructor is not constant and large.

program p
  implicit none
  call non_constant_bound()
  call large_array()
  print *, "PASS"
contains
  subroutine non_constant_bound()
    integer :: a(1, 100)
    integer :: b(50)
    integer :: m, n
    a = 1
    b = 2
    m = 55
    n = 10
    b(1:n) = [a(1:1, m:m+n-1)]
    if (any(b(1:n) /= 1)) STOP 1
  end subroutine

  subroutine large_array()
    integer :: a(1, 100)
    integer :: b(50)
    a = 1
    b = 2
    b = [a(1:1, 21:70)]
    if (any(b /= 1)) STOP 2
  end subroutine
end program
