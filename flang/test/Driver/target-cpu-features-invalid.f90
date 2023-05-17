! REQUIRES: aarch64-registered-target, amdgpu-registered-target

! Test that invalid cpu and features are ignored.

! RUN: %flang_fc1 -triple aarch64-linux-gnu -target-cpu supercpu \
! RUN:   -o /dev/null -S %s 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-CPU

! RUN: %flang_fc1 -triple aarch64-linux-gnu -target-feature +superspeed \
! RUN:   -o /dev/null -S %s 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-FEATURE

! RUN: not %flang_fc1 -triple amdgcn-amd-amdhsa -target-feature +wavefrontsize32 \
! RUN:   -target-feature +wavefrontsize64 -o /dev/null -S %s 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-WAVEFRONT

! CHECK-INVALID-CPU: 'supercpu' is not a recognized processor for this target (ignoring processor)
! CHECK-INVALID-FEATURE: '+superspeed' is not a recognized feature for this target (ignoring feature)
! CHECK-INVALID-WAVEFRONT: 'wavefrontsize32' and 'wavefrontsize64' are mutually exclusive
