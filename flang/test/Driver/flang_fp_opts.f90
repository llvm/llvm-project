! Test for handling of floating point options within the frontend driver

! RUN: %flang_fc1 -ffp-contract=fast %s 2>&1 | FileCheck %s
! CHECK: ffp-contract= is not currently implemented
