! Verify unknown, malformed, and -Werror= / -Wno-error= for another warning group.
!
! Unknown Flang groups fall through to Clang's ProcessWarningOptions and are
! accepted without "Unknown diagnostic option". Malformed -Werror= is rejected.
! Scanning warnings come from -ffixed-form on %S/Inputs/werror-unknown-scanning.f90.

!--- unknown Flang groups: no driver diagnostic --------------------------------
! RUN: %flang_fc1 -fsyntax-only -Werror=not-a-flang-warning %s 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-UNKNOWN
! RUN: %flang_fc1 -fsyntax-only -Wno-error=not-a-flang-warning %s 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-UNKNOWN

!--- malformed -Werror= spelling -----------------------------------------------
! RUN: not %flang_fc1 -fsyntax-only -Werror= %s 2>&1 | FileCheck %s --check-prefix=MALFORMED

!--- scanning warning group ----------------------------------------------------
! RUN: %flang_fc1 -fsyntax-only -ffixed-form %S/Inputs/werror-unknown-scanning.f90 -Werror -Wno-error=scanning 2>&1 | FileCheck %s --check-prefixes=SCAN-WARN,NO-ERROR
! RUN: not %flang_fc1 -fsyntax-only -ffixed-form %S/Inputs/werror-unknown-scanning.f90 -Werror=scanning 2>&1 | FileCheck %s --check-prefixes=SCAN-ERROR,SCAN-WARN
! RUN: not %flang_fc1 -fsyntax-only -ffixed-form %S/Inputs/werror-unknown-scanning.f90 -Werror 2>&1 | FileCheck %s --check-prefixes=SCAN-ERROR,SCAN-WARN

! NO-UNKNOWN-NOT: Unknown diagnostic option
! NO-UNKNOWN-NOT: not-a-flang-warning

! MALFORMED: Unknown diagnostic option: -Werror=

! FileCheck scans forward: SCAN-ERROR before SCAN-WARN.
! SCAN-ERROR: Could not scan
! SCAN-WARN: Statement should not begin with a continuation line [-Wscanning]
! NO-ERROR-NOT: error:

program werrorUnknown
end
