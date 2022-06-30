! Ensure the driver correctly switches between fixed and free form based on the file extension.
! This test exploits the fact that the prescanner treats whitespace differently for free
! and fixed form input files.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -E -Xflang -fno-reformat %S/Inputs/free-form-test.f90  2>&1 | FileCheck %s --check-prefix=FREEFORM
! RUN: %flang -E -Xflang -fno-reformat %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FIXEDFORM
! RUN: %flang -E -Xflang -fno-reformat %S/Inputs/free-form-test.f90 %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=MULTIPLEFORMS

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang_fc1)
!-----------------------------------------
! RUN: %flang_fc1 -E -fno-reformat %S/Inputs/free-form-test.f90  2>&1 | FileCheck %s --check-prefix=FREEFORM
! RUN: %flang_fc1 -E -fno-reformat %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FIXEDFORM
! RUN: %flang_fc1 -E -fno-reformat %S/Inputs/free-form-test.f90 %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=MULTIPLEFORMS

! FREEFORM:program freeform
! FREEFORM-NOT:programfixedform

! FIXEDFORM:programfixedform
! FIXEDFORM-NOT:program freeform

! MULTIPLEFORMS:program freeform
! MULTIPLEFORMS-NOT:programfixedform
! MULTIPLEFORMS-NEXT:end
! MULTIPLEFORMS-NEXT:programfixedform
! MULTIPLEFORMS-NOT:program freeform
