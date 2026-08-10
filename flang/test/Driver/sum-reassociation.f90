! Test driver handling of -fsum-reassociation and
! -fno-sum-reassociation, including the old hidden aliases.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fsum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-sum-reassociation -fsum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fsum-reassociation -fno-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -freal-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-real-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fsum-reassociation -fno-real-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-sum-reassociation -freal-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! DISABLED: "-fc1"
! DISABLED-NOT: "-fsum-reassociation"

! ENABLED: "-fc1"
! ENABLED-SAME: "-fsum-reassociation"
