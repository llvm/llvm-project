! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! bug in 1.7
! host association caused an error
!
module aa
 integer f
end module
module bb
 use aa
 contains
  subroutine jj(f)
   integer f
   f = f + 1
  end subroutine
end module

program p
 use aa
 use bb
 integer g
 integer result(2),expect(2)
 data expect/11,10/
 f = 9
 g = 10
 call jj(g)
 call jj(f)
 result(1) = g
 result(2) = f
 call check(result,expect,2)
end
