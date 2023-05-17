! Test that flang-new forwards -f{no-}color-diagnostics options to
! flang-new -fc1 as expected.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 -fcolor-diagnostics \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD
! CHECK-CD: "-fc1"{{.*}} "-fcolor-diagnostics"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 -fno-color-diagnostics \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD
! CHECK-NCD-NOT: "-fc1"{{.*}} "-fcolor-diagnostics"

! Check that the last flag wins.
! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-color-diagnostics -fcolor-diagnostics \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD_CD_S
! CHECK-NCD_CD_S: "-fc1"{{.*}} "-fcolor-diagnostics"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fcolor-diagnostics -fno-color-diagnostics \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD_NCD_S
! CHECK-CD_NCD_S-NOT: "-fc1"{{.*}} "-fcolor-diagnostics"
