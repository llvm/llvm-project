!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for argument "POSITION" of getarg with different kind

program test
  integer(kind=1) :: i1
  integer(kind=2) :: i2
  character(len=32) :: arg

  i1 = 1_1
  i2 = 2_2

  ! the commond line arguments in ".mk" file is:
  ! $(TEST).$(EXESUFFIX) a b

  call getarg(i1,arg)
  if (arg /= 'a') STOP 1
  call getarg(i2,arg)
  if (arg /= 'b') STOP 2

  print *, 'PASS'
end program
