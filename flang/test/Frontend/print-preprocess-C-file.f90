! Test preprocessing for C files using Flang driver

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang-new -E %S/Inputs/hello-world.c  2>&1 | FileCheck %s

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! CHECK: error: unknown integrated tool '-cc1'. Valid tools include '-fc1'.
