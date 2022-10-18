! Test for handling of floating point options within the frontend driver

! RUN: %flang_fc1 \
! RUN:      -ffp-contract=fast \
! RUN:      -menable-no-infs \
! RUN:      -menable-no-nans \
! RUN:      -fapprox-func \
! RUN:      %s 2>&1 | FileCheck %s
! CHECK: ffp-contract= is not currently implemented
! CHECK: menable-no-infs is not currently implemented
! CHECK: menable-no-nans is not currently implemented
! CHECK: fapprox-func is not currently implemented
