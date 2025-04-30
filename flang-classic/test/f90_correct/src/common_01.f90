!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Check if two common block variables are equal when turn on optimization

module mod1
  integer :: mi1
  common /com1/ mi1
contains
  subroutine newmi1()
    mi1 = 30
  end subroutine newmi1
end module mod1

program test
  use mod1
  use check_mod
  integer, parameter :: n = 6
  logical :: reslt(n), expct(n)
  integer :: pi1
  common /com1/ pi1

  reslt = .false.
  expct = .true.
  pi1 = 10
  mi1 = 20
  reslt(1:3) = (/pi1 == mi1, pi1 == 20, mi1 == 20/)
  call newmi1()
  reslt(4:6) = (/pi1 == mi1, pi1 == 30, mi1 == 30/)

  call check(reslt, expct, n)
end
