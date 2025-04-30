!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program implied_do11
  implicit none
  real :: x(2) = 0
  real :: y(3) = 0
  real :: z(10) = 0
  integer :: i

  x = (/ (f1(1), i = 1, 2) /)
  if (any(x /= 4.0)) STOP 1
  y = (/ (f1(i), i = 1, 2) /)
  if (any(y /= (/4.0, 8.0, 8.0/))) STOP 2
  z = (/ (f1(5), i = 1, 2) /)
  if (any(z /= 20.0)) STOP 3
  x = (/ (f2(1), i = 1, 2) /)
  if (any(x /= 10.0)) STOP 4
  y = (/ (f2(i), i = 1, 2) /)
  if (any(y /= (/10.0, 20.0, 20.0/))) STOP 5
  z = (/ (f2(5), i = 1, 2) /)
  if (any(z /= 50.0)) STOP 6
  print *, 'PASS'
contains
  function f1(n)
    implicit none
    integer :: n
    real :: f1(n)

    f1 = n * 4.0
  end function
  function f2(n)
    implicit none
    integer :: n
    real, pointer :: f2(:)
    allocate(f2(n))

    f2 = n * 10.0
  end function

end program
