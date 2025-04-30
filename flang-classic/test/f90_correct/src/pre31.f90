!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests super-long (Fortran line length violating) comments that should be
! passed as multiple lines (having the preprocessor not JOIN new lines in these
! comments).
!
!!                     &4 byte REAL     &8 byte REAL        \\\
!!        CRAY:        &-               &precision =   13   \\\
!!                     &                &exponent  = 2465   \\\
!!        IEEE:        &precision = 6   &precision =   15   \\\
!!                     &exponent  = 37  &exponent  =  307
program p
    call check(.true., .true., 1)
end program
