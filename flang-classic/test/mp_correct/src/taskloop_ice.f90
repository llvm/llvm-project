!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
program reproducer_taskloop
    integer, parameter :: n=10
    integer :: expect(n) = (/ 1,2,3,4,5,6,7,8,9,10 /)
    integer :: result(n)

    integer :: i
    result = 2
    !$OMP TASKLOOP private(i) shared(result)
    do i = 1, 10
      result(i) = i
    end do
    !$OMP END TASKLOOP

    call check(result,expect,n)
end program reproducer_taskloop
