! Ensure argument -Werror work as expected, this file checks for the functional correctness for
! actions that extend the PrescanAndSemaAction, particularly for Semantic warnings/errors.
! Multiple RUN lines are added to make sure that the behavior is consistent across multiple actions.

! RUN: not %flang_fc1 -fsyntax-only -std=f2018 -Werror %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -std=f2018 -Werror -fdebug-dump-parse-tree %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -std=f2018 -Werror -fdebug-unparse-with-symbols %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -std=f2018 -Werror -fdebug-unparse %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -std=f2018 -Werror -fdebug-dump-symbols %s  2>&1 | FileCheck %s --check-prefix=WITH


! RUN: %flang_fc1 -fsyntax-only -std=f2018 %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -std=f2018 -fdebug-dump-parse-tree %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -std=f2018 -fdebug-unparse-with-symbols %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -std=f2018 -fdebug-unparse %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -std=f2018 -fdebug-dump-symbols %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT

!-----------------------------------------
! EXPECTED OUTPUT WITH -Werror
!-----------------------------------------
! WITH: Could not parse

!-----------------------------------------
! EXPECTED OUTPUT WITHOUT -Werror
!-----------------------------------------
! WITHOUT-NOT: Could not parse

#ifndef _OM_NO_IOSTREAM
#ifdef WIN32
#ifndef USE_IOSTREAM
#define USE_IOSTREAM
#endif USE_IOSTREAM
#endif WIN32
