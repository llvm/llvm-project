!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check that loops in containing routines don't update counter in container

program p
 integer i,j,k
 integer result(4)
 integer expect(4)
 data expect/5,50,15,150/
 j = 0
 result(2) = 0
 do i = 1,5
  j = j + 1
  call sub(k)
  result(2) = result(2) + k
 enddo
 result(1) = j
 j = 0
 result(4) = 0
 do i = 1,15
  j = j + 1
  call sub(k)
  result(4) = result(4) + k
 enddo
 result(3) = j
!print *,result
 call check(result,expect,4)

contains
 subroutine sub(k)
  integer i,k
  k = 0
  do i = 1,10
   k = k + 1
  enddo
 end subroutine
end program
