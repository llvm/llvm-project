!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test the fix for introduced error, in which the entry returns adjustable
! length character.
module m
contains
function f1() bind(C)
  real (kind=4) :: f1
  integer :: e1
  f1 = 8.0
  return
entry e1() bind(C)
  e1 = 4
end function
end module

program test
  use m
  real(kind=4) :: resc
  integer :: resi,res2(1),exp2(1)
  real(kind=4) :: res1(1),exp1(1)

  resc=f1()
  resi = e1()
  print *, "hello", resc, resi
  res1(1)= resc
  exp1(1)=8.0

  res2 = resi;
  exp2(1)=4
  call checkf(res1,exp1,1)
  call check(res2,exp2,1)
end program
