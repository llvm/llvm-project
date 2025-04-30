! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for intrinsic funciont take quad argument.

program test_cpu_time
    real(16) :: start, finish
    integer :: i, n
    call cpu_time(start)
    ! code to delay times.
    do i = 1, 10000
      n = i ** 1.1
    end do
    call cpu_time(finish)
    if (finish - start <= 0) STOP 1
    ! print n to avoid being optimized in O2/O3.
    print *, n
    print *, 'PASS'
end program test_cpu_time

