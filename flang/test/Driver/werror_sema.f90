! Ensure argument -Werror work as expected, this file checks for the functional correctness for
! actions that extend the PrescanAndSemaAction, particularly for Semantic warnings/errors.
! Multiple RUN lines are added to make sure that the behavior is consistent across multiple actions.

! RUN: not %flang_fc1 -fsyntax-only -Werror %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -fsyntax-only -Werror -fdebug-dump-parse-tree %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -fsyntax-only -Werror -fdebug-unparse-with-symbols %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -fsyntax-only -Werror -fdebug-unparse %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -fsyntax-only -Werror -fdebug-dump-symbols %s  2>&1 | FileCheck %s --check-prefix=WITH


! RUN: %flang_fc1 -fsyntax-only %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fsyntax-only -fdebug-dump-parse-tree %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fsyntax-only -fdebug-unparse-with-symbols %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fsyntax-only -fdebug-unparse %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fsyntax-only -fdebug-dump-symbols %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT

!-----------------------------------------
! EXPECTED OUTPUT WITH -Werror
!-----------------------------------------
! WITH: Semantic errors in

!-----------------------------------------
! EXPECTED OUTPUT WITHOUT -Werror
!-----------------------------------------
! WITHOUT-NOT: Semantic errors in

PROGRAM werror
REAL, DIMENSION(20, 10) :: A
FORALL (J=1:N)  A(I, I) = 1
END PROGRAM werror
