! Test driver handling of -freal-sum-reassociation and
! -fno-real-sum-reassociation.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -freal-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-real-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-real-sum-reassociation -freal-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -freal-sum-reassociation -fno-real-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! DISABLED: "-fc1"
! DISABLED-NOT: "-freal-sum-reassociation"

! ENABLED: "-fc1"
! ENABLED-SAME: "-freal-sum-reassociation"
