!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests the warning and line directives.  The warning should be listed as
! occurring on line 43 since the #line directive should have adjusted the line
! offset.
!
! This tests 'C' in a macro proceeding a directive (it is not a comment)
!
#define C "A string"
program p
#ifdef C
    print *, C
    print *, "PASS"
    call check(.true., .true., 1)
#else
    print *, "FAIL"
    call check(.false., .true., 1)
#endif
end program
