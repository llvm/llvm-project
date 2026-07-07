! Test driver handling of -freal-sum-reassociation and
! -fno-real-sum-reassociation.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DEFAULT
! DEFAULT: "-fc1"
! DEFAULT-NOT: "-freal-sum-reassociation"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -freal-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLE
! ENABLE: "-fc1"{{.*}} "-freal-sum-reassociation"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-real-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLE
! DISABLE: "-fc1"
! DISABLE-NOT: "-freal-sum-reassociation"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-real-sum-reassociation -freal-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=ENABLE-LAST
! ENABLE-LAST: "-fc1"{{.*}} "-freal-sum-reassociation"

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -freal-sum-reassociation -fno-real-sum-reassociation \
! RUN:   | FileCheck %s --check-prefix=DISABLE-LAST
! DISABLE-LAST: "-fc1"
! DISABLE-LAST-NOT: "-freal-sum-reassociation"
