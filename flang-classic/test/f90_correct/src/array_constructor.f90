!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module array_constructor_module
  implicit none

  integer, private :: i

  integer, parameter :: some_array(3) = (/ (i, i = 4241, 4243) /)

end module array_constructor_module

program test
  use array_constructor_module
  implicit none

  integer, parameter :: num = 1
  integer rslts(num), expect(num)
  data expect / 12726 /

  ! test that summing elements of an array where the array lives in another
  ! module produces the correct result.
  rslts(1) = SUM(some_array)
  call check(rslts, expect, num)

end program
