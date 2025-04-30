!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests a macro and a fixed-form comment (both begin with 'C')
! This should be run with pgf90:
!    -Mpreprocess
!    -mp
!    -Hx,124,0x100000 (Do not perform macro expansion of comments, as
!                      that would replace 'C a comment' line below with '42 a
!                      comment'
!    -Mfixed
!
#define C 42
C a comment
      program p
C$omp FLUSH
            if (C .eq. 42) then
                call check(.true., .true., 1)
            else
                call check(.false., .true., 1)
            endif
      end
