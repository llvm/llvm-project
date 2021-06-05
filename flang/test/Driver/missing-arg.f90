! Make sure that frontend driver options that require arguments are
! correctly rejected when the argument value is missing.

! REQUIRES: new-flang-driver

!-----------
! RUN lines
!-----------
! RUN: not %flang_fc1 -E %s -o 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -E %s -U 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -E %s -D 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -E %s -I 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -E %s -J 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -E %s -module-dir 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -E %s -module-suffix 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -E %s -fintrinsic-modules-path 2>&1 | FileCheck %s

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! CHECK: error: argument to '-{{.*}}' is missing (expected 1 value)
