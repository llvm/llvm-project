! Test that flang forwards -fdiagnostics-absolute-paths to flang -fc1 as
! expected.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 -fdiagnostics-absolute-paths \
! RUN:   | FileCheck %s --check-prefix=CHECK-AP
! CHECK-AP: "-fc1"{{.*}} "-fdiagnostics-absolute-paths"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NO-AP
! CHECK-NO-AP-NOT: "-fc1"{{.*}} "-fdiagnostics-absolute-paths"
