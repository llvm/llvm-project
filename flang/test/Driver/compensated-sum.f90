! Test driver handling of -fcompensated-sum.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=OMITTED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fcompensated-sum \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! OMITTED: "-fc1"
! OMITTED-NOT: "-fcompensated-sum"

! ENABLED: "-fc1"
! ENABLED: "-fcompensated-sum"
