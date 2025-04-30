! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!   Internal procedures
!   don't delete assignments to outer-block variables

program p

integer i,j,k
integer result(3),expect(3)
data expect/99,199,299/

i = 100
j = 98
k = 299		! k not used, but must not be deleted
call sub
result(1)=i	! i must not be forward-substituted
result(2)=j	! j must not be forward-substituted
call check(result,expect,3)

contains

subroutine sub

i = i - 1	! i never used, should not be deleted
j = 199		! j never used, should not be deleted
result(3) = k	! k comes from outer block

end subroutine sub
end program p
