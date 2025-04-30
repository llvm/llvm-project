! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program p
implicit none
logical rslt(2),expect(2)
type :: objects(l1)
integer, len :: l1 = 10
character(len=l1) :: c
end type


type(objects(5)) :: x

expect = .true.

x%c = '12345'

rslt(1) = x%c .eq. '12345'
rslt(2) = len_trim(x%c) .eq. x%l1

call check(rslt,expect,2)
end



