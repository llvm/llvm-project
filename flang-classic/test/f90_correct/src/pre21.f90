!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests $ and ',' restrictions on macro identifiers and in directives
!
#define A$B "Hello"

program p
#if defined(A$B), 0x11 > 0x10
    print *, A$B
    call check(.true., .true., 1)
#else
    print *, "FAIL"
    call check(.false., .true., 1)
#endif
end program
