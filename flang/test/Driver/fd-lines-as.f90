! Ensure arguments -fd-lines-as-comments and -fd-lines-as-code as expected.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! Default behavior is equivalent as -fd-lines-as-comments
!--------------------------
! RUN: %flang -fsyntax-only -ffixed-form %s 2>&1
! RUN: %flang -fsyntax-only -ffixed-form -fd-lines-as-comments %s 2>&1
! RUN: not %flang -fsyntax-only -ffixed-form -fd-lines-as-code %s 2>&1 | FileCheck %s --check-prefix=CODE
! RUN: not %flang -fsyntax-only -ffree-form -fd-lines-as-comments %s 2>&1 | FileCheck %s --check-prefix=WARNING-COMMENTS
! RUN: not %flang -fsyntax-only -ffree-form -fd-lines-as-code %s 2>&1 | FileCheck %s --check-prefix=WARNING-CODE

!----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!----------------------------------------
! RUN: %flang_fc1 -fsyntax-only -ffixed-form %s 2>&1
! RUN: %flang_fc1 -fsyntax-only -ffixed-form -fd-lines-as-comments %s 2>&1
! RUN: not %flang_fc1 -fsyntax-only -ffixed-form -fd-lines-as-code %s 2>&1 | FileCheck %s --check-prefix=CODE
! RUN: not %flang_fc1 -fsyntax-only -ffree-form -fd-lines-as-comments %s 2>&1 | FileCheck %s --check-prefix=WARNING-COMMENTS
! RUN: not %flang_fc1 -fsyntax-only -ffree-form -fd-lines-as-code %s 2>&1 | FileCheck %s --check-prefix=WARNING-CODE

! CODE: Semantic errors
! WARNING-COMMENTS: warning: ‘-fd-lines-as-comments’ has no effect in free form.
! WARNING-CODE: warning: ‘-fd-lines-as-code’ has no effect in free form.

      program FixedForm
d     end
      end
