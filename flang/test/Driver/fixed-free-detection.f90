! Ensure the driver correctly switches between fixed and free form based on the file extension.
! This test exploits the fact that the preprocessor treats white-spaces differently for free
! and fixed form input files.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -E %S/Inputs/free-form-test.f90  2>&1 | FileCheck %s --check-prefix=FREEFORM
! RUN: %flang -E %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FIXEDFORM
! RUN: %flang -E %S/Inputs/free-form-test.f90 %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=MULTIPLEFORMS

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang_fc1)
!-----------------------------------------
! RUN: %flang_fc1 -E %S/Inputs/free-form-test.f90  2>&1 | FileCheck %s --check-prefix=FREEFORM
! RUN: %flang_fc1 -E %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FIXEDFORM
! RUN: %flang_fc1 -E %S/Inputs/free-form-test.f90 %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=MULTIPLEFORMS

!-------------------------------------
! EXPECTED OUTPUT FOR A FREE FORM FILE
!-------------------------------------
! FREEFORM:program freeform
! FREEFORM-NOT:programfixedform

!---------------------------------------
! EXPECTED OUTPUT FOR A FIXED FORM FILE
!---------------------------------------
! FIXEDFORM:programfixedform
! FIXEDFORM-NOT:program freeform

!------------------------------------------------
! EXPECTED OUTPUT FOR 2 FILES OF DIFFERENT FORMS
!------------------------------------------------
! MULTIPLEFORMS:program freeform
! MULTIPLEFORMS-NOT:programfixedform
! MULTIPLEFORMS-NEXT:end
! MULTIPLEFORMS-NEXT:programfixedform
! MULTIPLEFORMS-NOT:program freeform
