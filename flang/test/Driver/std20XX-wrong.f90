! Ensure argument -std=f20XX works as expected.

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!-----------------------------------------
! RUN: not %flang_fc1 -std=90 %s  2>&1 | FileCheck %s --check-prefix=WRONG

! WRONG: Only 'f2018', 'f2023', or 'f202Y' are accepted to -std= currently.
