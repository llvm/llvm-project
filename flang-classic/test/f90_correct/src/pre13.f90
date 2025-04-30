!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests line, file, date, and time directives,
! as well as the '#\n' (null directive)
!
#line 42 "universe.f90"
#
program p
    print *, __FILE__
    print *, __LINE__
    print *, __DATE__
    print *, __TIME__
    call check(.true., .true., 1)
end program
