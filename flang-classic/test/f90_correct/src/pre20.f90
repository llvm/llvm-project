!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests that the preprocessor does not detect trigraph-like sequences...
! this is Fortran not C
!
program p
    print *, "Four quotes:", '????'
    print *, "Four quotes:", "????"
    call check(.true., .true., 1)
end program
