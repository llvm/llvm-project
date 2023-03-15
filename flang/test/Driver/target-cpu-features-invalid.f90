! REQUIRES: aarch64-registered-target

! Test that invalid cpu and features are ignored.

! RUN: %flang_fc1 -triple aarch64-linux-gnu -target-cpu supercpu \
! RUN:   -o /dev/null -S %s 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-CPU

! RUN: %flang_fc1 -triple aarch64-linux-gnu -target-feature +superspeed \
! RUN:   -o /dev/null -S %s 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-FEATURE


! CHECK-INVALID-CPU: 'supercpu' is not a recognized processor for this target (ignoring processor)
! CHECK-INVALID-FEATURE: '+superspeed' is not a recognized feature for this target (ignoring feature)
