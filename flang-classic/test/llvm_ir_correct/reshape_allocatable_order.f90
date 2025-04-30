!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -S -emit-llvm %s

program test_reshape
  integer, dimension(4) :: x = [1, 2, 3, 4]
  integer, dimension(:), allocatable :: order
  integer, dimension(2) :: other_order = [2, 1]
  allocate(order(2))
  order(:) = [2, 1]
  print *, reshape(x, shape = [2, 2], order = order)
!!  print *, reshape(x, shape = [2, 2], order = other_order)
end program
