!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test the fix for regression introduced by PR!1101.

module m
  implicit none
  type empty
  end type empty
  type(empty) :: x = empty()
  type(empty), parameter :: y = empty()
end module m

program test
  use m
  implicit none
  print *, "PASS"
end program
