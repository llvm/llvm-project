! Test that flang-new forwards -f{no-}color-diagnostics and
! -f{no-}diagnostics-color options to flang-new -fc1 as expected.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 -fcolor-diagnostics \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD
! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 -fdiagnostics-color \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD
! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 -fdiagnostics-color=always \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD
! CHECK-CD: "-fc1"{{.*}} "-fcolor-diagnostics"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 -fno-color-diagnostics \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD
! RUN: %flang -fsyntax-only -### %s -o %t -fno-diagnostics-color 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD
! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 -fdiagnostics-color=never \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD
! CHECK-NCD-NOT: "-fc1"{{.*}} "-fcolor-diagnostics"

! Check that the last flag wins.
! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-color-diagnostics -fcolor-diagnostics \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD_CD_S
! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-diagnostics-color -fdiagnostics-color \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD_CD_S
! RUN: %flang -fsyntax-only -### %s -o %t \
! RUN:     -fno-color-diagnostics -fdiagnostics-color=always 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD_CD_S
! RUN: %flang -fsyntax-only -### %s -o %t \
! RUN:     -fdiagnostics-color=never -fdiagnostics-color=always 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD_CD_S
! RUN: %flang -fsyntax-only -### %s -o %t \
! RUN:     -fdiagnostics-color=never -fcolor-diagnostics 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NCD_CD_S
! CHECK-NCD_CD_S: "-fc1"{{.*}} "-fcolor-diagnostics"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fcolor-diagnostics -fno-color-diagnostics \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD_NCD_S
! RUN: %flang -fsyntax-only -### %s -o %t \
! RUN:     -fdiagnostics-color -fno-diagnostics-color  2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD_NCD_S
! RUN: %flang -fsyntax-only -### %s -o %t \
! RUN:     -fdiagnostics-color=always -fno-color-diagnostics 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD_NCD_S
! RUN: %flang -fsyntax-only -### %s -o %t \
! RUN:     -fdiagnostics-color=always -fdiagnostics-color=never 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD_NCD_S
! RUN: %flang -fsyntax-only -### %s -o %t \
! RUN:     -fcolor-diagnostics -fdiagnostics-color=never 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-CD_NCD_S
! CHECK-CD_NCD_S-NOT: "-fc1"{{.*}} "-fcolor-diagnostics"
