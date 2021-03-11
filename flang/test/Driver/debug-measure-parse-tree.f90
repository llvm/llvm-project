! Ensure argument -fdebug-measure-parse-tree works as expected.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang-new -fdebug-measure-parse-tree %s  2>&1 | FileCheck %s --check-prefix=FLANG

!----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!----------------------------------------
! RUN: %flang-new -fc1 -fdebug-measure-parse-tree %s  2>&1 | FileCheck %s --check-prefix=FRONTEND

!----------------------------------
! EXPECTED OUTPUT WITH `flang-new`
!----------------------------------
! FLANG:warning: argument unused during compilation: '-fdebug-measure-parse-tree'

!---------------------------------------
! EXPECTED OUTPUT WITH `flang-new -fc1`
!---------------------------------------
! FRONTEND:Parse tree comprises {{[0-9]+}} objects and occupies {{[0-9]+}} total bytes.

program A
end
