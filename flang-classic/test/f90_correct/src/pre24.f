!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests comments and a macro beginning with 'C'
!
#define COOL "C is for cookie"
C This is a comment
      program p
      if (COOL == "C is for cookie") then
            call check(.true., .true., 1)
      else
            call check(.false., .true., 1)
      endif
      print *, COOL
      end
