! Ensure arguments -ffree-form and -ffixed-form work as expected.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang-new -fsyntax-only -ffree-form %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FREEFORM
! RUN: %flang-new -fsyntax-only -ffixed-form %S/Inputs/free-form-test.f90 2>&1 | FileCheck %s --check-prefix=FIXEDFORM

!----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!----------------------------------------
! RUN: not %flang-new -fc1 -fsyntax-only -ffree-form %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FREEFORM
! RUN: %flang-new -fc1 -fsyntax-only -ffixed-form %S/Inputs/free-form-test.f90 2>&1 | FileCheck %s --check-prefix=FIXEDFORM

!------------------------------------
! EXPECTED OUTPUT FOR FREE FORM MODE
!------------------------------------
! FREEFORM:error: Could not parse

!-------------------------------------
! EXPECTED OUTPUT FOR FIXED FORM MODE
!-------------------------------------
! FIXEDFORM:free-form-test.f90:1:1: Character in fixed-form label field must be a digit
