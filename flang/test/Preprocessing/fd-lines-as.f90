! Ensure arguments -fd-lines-as-comments and -fd-lines-as-code display the correct output with -E.

!--------------------------
! Default behavior is equivalent as -fd-lines-as-comments
!--------------------------
! RUN: %flang -E -ffixed-form %s 2>&1 | FileCheck %s --check-prefix=COMMENT
! RUN: %flang -E -ffixed-form -fd-lines-as-comments %s 2>&1 | FileCheck %s --check-prefix=COMMENT
! RUN: %flang -E -ffixed-form -fd-lines-as-code %s 2>&1 | FileCheck %s --check-prefix=CODE

      program FixedForm
d     end
D     end
      end

! COMMENT: program FixedForm
! COMMENT: end
! COMMENT-NOT: end
! COMMENT-NOT: end

! CODE: program FixedForm
! CODE: end
! CODE: end
! CODE: end
