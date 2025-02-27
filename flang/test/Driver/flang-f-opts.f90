! Test for warnings generated when parsing driver options. You can use this file for relatively small tests and to avoid creating
! new test files.

! RUN: %flang -### -S -O4 -ffp-contract=on %s 2>&1 | FileCheck %s

! CHECK: warning: the argument 'on' is not supported for option 'ffp-contract='. Mapping to 'ffp-contract=off'
! CHECK: warning: -O4 is equivalent to -O3
! CHECK-LABEL: "-fc1"
! CHECK: -ffp-contract=off
! CHECK: -O3
