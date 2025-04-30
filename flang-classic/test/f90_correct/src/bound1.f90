!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND in type specification.

program test
  implicit none
  integer :: x(2,3,4)
  integer, allocatable :: y(:)
  integer :: i

  y = foo(x)
  if (size(y) /= 24 .or. any(y /= 1)) STOP 1
  y = bar(x)
  if (size(y) /= 3 .or. any(y /= 2)) STOP 2
  print *, "PASS"
contains
  function foo(a)
    integer :: a(:, :, :)
    integer :: foo(1:product(ubound(a)))
    foo = 1
  end function

  function bar(a)
    integer :: a(:, :, :)
    integer :: bar(1:sum(lbound(a)))
    bar = 2
  end function
end program
