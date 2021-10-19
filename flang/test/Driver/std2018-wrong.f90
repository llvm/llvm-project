! Ensure argument -std=f2018 works as expected.

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: not %flang_fc1 -std=90 %s  2>&1 | FileCheck %s --check-prefix=WRONG

!-----------------------------------------
! EXPECTED OUTPUT WITH WRONG
!-----------------------------------------
! WRONG: Only -std=f2018 is allowed currently.
