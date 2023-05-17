! Ensure argument -Werror work as expected, this file checks for the functional correctness for
! actions that extend the PrescanAndSemaAction, particularly for Semantic warnings/errors.
! Multiple RUN lines are added to make sure that the behavior is consistent across multiple actions.

! RUN: not %flang_fc1 -fsyntax-only -Werror %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -Werror -fdebug-dump-parse-tree %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -Werror -fdebug-unparse-with-symbols %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -Werror -fdebug-unparse %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -Werror -fdebug-dump-symbols %s  2>&1 | FileCheck %s --check-prefix=WITH


! RUN: %flang_fc1 -fsyntax-only %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fdebug-dump-parse-tree %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fdebug-unparse-with-symbols %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fdebug-unparse %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fdebug-dump-symbols %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT

! WITH: Semantic errors in

! WITHOUT-NOT: Semantic errors in

PROGRAM werror
REAL, DIMENSION(20, 10) :: A
FORALL (J=1:N)  A(I, I) = 1
END PROGRAM werror
