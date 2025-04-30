!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Parallel workshare OpenMP test
!


program test
integer :: res = 0
integer :: expected = 1
call try1(res)
call check(res, expected, 1)
stop
end


subroutine try1(x)
integer :: x

!$OMP PARALLEL WORKSHARE
x = 1
!$OMP END PARALLEL WORKSHARE

end subroutine try1

