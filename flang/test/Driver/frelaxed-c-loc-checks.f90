! Check that -frelaxed-c-loc-checks is forwarded from the driver to fc1.
! RUN: %flang -### -frelaxed-c-loc-checks %s 2>&1 | FileCheck %s
! CHECK: "-fc1"
! CHECK-SAME: "-frelaxed-c-loc-checks"
