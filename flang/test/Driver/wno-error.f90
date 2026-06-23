! Verify -Werror / -Werror= / -Wno-error / -Wno-error= for Flang warning groups.
!
! Portability warning from -pedantic on ichar('ab').
! Redundant-attribute warning from duplicate SAVE on integer, save, save :: x.
! Use -Wno-redundant-attribute on portability-only runs so they stay isolated.

!--- Baseline (no -pedantic) --------------------------------------------------
! RUN: %flang_fc1 -fsyntax-only -Wno-redundant-attribute %s 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-DIAG

!--- -Werror=portability without -pedantic: no warning, so no error -----------
! RUN: %flang_fc1 -fsyntax-only -Wno-redundant-attribute -Werror=portability %s 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-DIAG

!--- -pedantic and redundant-attribute both emit warnings --------------------
! RUN: %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s --check-prefixes=PEDANTIC-WARN,REDUNDANT-WARN,NO-ERROR

!--- combinations that promote the portability warning to an error ------------
! RUN: not %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Werror %s 2>&1 | FileCheck %s --check-prefixes=IS-ERROR,PEDANTIC-WARN
! RUN: not %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Werror=portability %s 2>&1 | FileCheck %s --check-prefixes=IS-ERROR,PEDANTIC-WARN
! RUN: not %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Wno-error -Werror=portability %s 2>&1 | FileCheck %s --check-prefixes=IS-ERROR,PEDANTIC-WARN
! RUN: not %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Werror -Wno-error -Werror=portability %s 2>&1 | FileCheck %s --check-prefixes=IS-ERROR,PEDANTIC-WARN

!--- combinations that leave the portability warning non-fatal ----------------
! RUN: %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Wno-error %s 2>&1 | FileCheck %s --check-prefixes=PEDANTIC-WARN,NO-ERROR
! RUN: %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Wno-error=portability %s 2>&1 | FileCheck %s --check-prefixes=PEDANTIC-WARN,NO-ERROR
! RUN: %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Werror -Wno-error=portability %s 2>&1 | FileCheck %s --check-prefixes=PEDANTIC-WARN,NO-ERROR
! RUN: %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Werror -Wno-error %s 2>&1 | FileCheck %s --check-prefixes=PEDANTIC-WARN,NO-ERROR
! RUN: %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Werror=portability -Wno-error=portability %s 2>&1 | FileCheck %s --check-prefixes=PEDANTIC-WARN,NO-ERROR
! RUN: %flang_fc1 -fsyntax-only -pedantic -Wno-redundant-attribute -Wno-error=portability -Werror %s 2>&1 | FileCheck %s --check-prefixes=PEDANTIC-WARN,NO-ERROR

!--- per-group control with both portability and redundant-attribute warnings --
! RUN: not %flang_fc1 -fsyntax-only -pedantic -Werror -Wno-error=portability %s 2>&1 | FileCheck %s --check-prefixes=IS-ERROR,REDUNDANT-WARN
! RUN: not %flang_fc1 -fsyntax-only -pedantic -Werror -Wno-error=redundant-attribute %s 2>&1 | FileCheck %s --check-prefixes=IS-ERROR,PEDANTIC-WARN
! RUN: %flang_fc1 -fsyntax-only -pedantic -Werror -Wno-error=portability -Wno-error=redundant-attribute %s 2>&1 | FileCheck %s --check-prefixes=PEDANTIC-WARN,REDUNDANT-WARN,NO-ERROR
! RUN: not %flang_fc1 -fsyntax-only -pedantic -Werror=redundant-attribute -Wno-error=portability %s 2>&1 | FileCheck %s --check-prefixes=IS-ERROR,REDUNDANT-WARN
! RUN: not %flang_fc1 -fsyntax-only -pedantic -Werror=portability -Wno-error=redundant-attribute %s 2>&1 | FileCheck %s --check-prefixes=IS-ERROR,PEDANTIC-WARN

! NO-DIAG-NOT: portability
! NO-DIAG-NOT: warning
! NO-DIAG-NOT: error

! FileCheck scans forward: IS-ERROR before PEDANTIC-WARN; REDUNDANT-WARN before
! PEDANTIC-WARN when both warnings appear (redundant-attribute is emitted first).
! IS-ERROR: Semantic errors in
! REDUNDANT-WARN: Attribute 'SAVE' cannot be used more than once [-Wredundant-attribute]
! PEDANTIC-WARN: should have length one [-Wportability]
! NO-ERROR-NOT: error:

subroutine wnoErrorTest
  integer, save, save :: x
  x = 1
  print *, ichar('ab')
end
