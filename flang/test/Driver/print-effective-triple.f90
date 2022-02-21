! Test that -print-target-triple prints correct triple.

!-----------------------------------------
! RUN LINE
!-----------------------------------------
! RUN: %flang -print-effective-triple 2>&1 --target=thumb-linux-gnu | FileCheck %s

!-----------------
! EXPECTED OUTPUT
!-----------------
! CHECK: armv4t-unknown-linux-gnu
