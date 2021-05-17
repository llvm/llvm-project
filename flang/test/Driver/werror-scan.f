! Ensure argument -Werror work as expected, this file checks for the functional correctness for
! actions that extend the PrescanAction
! Multiple RUN lines are added to make sure that the behavior is consistent across multiple actions.

! RUN: not %flang_fc1 -fsyntax-only -E -Werror %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -fsyntax-only -fdebug-dump-parsing-log -Werror %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -fsyntax-only -fdebug-dump-provenance -Werror %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: not %flang_fc1 -fsyntax-only -fdebug-measure-parse-tree -Werror %s  2>&1 | FileCheck %s --check-prefix=WITH
! RUN: %flang_fc1 -fsyntax-only -E %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fsyntax-only -fdebug-dump-parsing-log %s  2>&1 | FileCheck %s --check-prefix=WITHOUT
! RUN: %flang_fc1 -fsyntax-only -fdebug-dump-provenance %s  2>&1 | FileCheck %s --check-prefix=WITHOUT
! RUN: %flang_fc1 -fsyntax-only -fdebug-measure-parse-tree %s  2>&1 | FileCheck %s --check-prefix=WITHOUT

!-----------------------------------------
! EXPECTED OUTPUT WITH -Werror
!-----------------------------------------
! WITH: Could not scan

!-----------------------------------------
! EXPECTED OUTPUT WITHOUT -Werror
!-----------------------------------------
! WITHOUT-NOT: Could not scan

1 continue
end
