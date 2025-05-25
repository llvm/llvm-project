! Test for correct forwarding of integer overflow flags from the compiler driver
! to the frontend driver

! RUN: %flang -### -fno-strict-overflow %s 2>&1 | FileCheck %s --check-prefix=INDUCED
! RUN: %flang -### -fstrict-overflow %s 2>&1 | FileCheck %s
! RUN: %flang -### -fno-wrapv %s 2>&1 | FileCheck %s
! RUN: %flang -### -fno-wrapv -fno-strict-overflow %s 2>&1 | FileCheck %s

! CHECK-NOT: "-fno-wrapv"
! INDUCED: "-fwrapv"
