! Test driver handling of -ffp-sum-reassociation and
! -fno-fp-sum-reassociation, including the old hidden aliases.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=OMITTED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -ffp-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-fp-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-fp-sum-reassociation -ffp-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -ffp-sum-reassociation -fno-fp-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -freal-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-real-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -ffp-sum-reassociation -fno-real-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-fp-sum-reassociation -freal-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! OMITTED: "-fc1"
! OMITTED-NOT: "-ffp-sum-reassociation"
! OMITTED-NOT: "-fno-fp-sum-reassociation"

! DISABLED: "-fc1"
! DISABLED-SAME: "-fno-fp-sum-reassociation"
! DISABLED-NOT: "-ffp-sum-reassociation"

! ENABLED: "-fc1"
! ENABLED-SAME: "-ffp-sum-reassociation"
! ENABLED-NOT: "-fno-fp-sum-reassociation"
