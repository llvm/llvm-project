!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

  program p
   integer results(9), expect(9)
   data expect / 12,12,13,22,22,23,32,32,33 /
   type ttype
    integer, dimension(3) :: X
   end type
   type(ttype), dimension(3) :: var
   integer i,j,k
   do i = 1,3
    do j = 1,3
     var(i)%x(j) = i*10+j
    enddo
   enddo
   i = 1
   j = 2
   var%X(i) = var%X(j)
   do i = 1,3
    do j = 1,3
     k = (i-1)*3+j
     results(k) = var(i)%X(j)
    enddo
   enddo
  call check(results,expect,9)
  END
