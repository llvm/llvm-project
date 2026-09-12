! Test -fdisable-real-16 works as expected.

! RUN: %flang -### -c %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-OPTION

! RUN: %flang -### -c -fdisable-real-16 %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DISABLE

! NO-OPTION-NOT: -fdisable-real-16
! DISABLE: -fdisable-real-16
