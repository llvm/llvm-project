! REQUIRES: amdgpu-registered-target

! Test that -mcpu are used and that the -target-cpu and -target-features
! are also added to the fc1 command.

! RUN: %flang --target=amdgcn-amd-amdhsa -mcpu=gfx902  -nogpulib -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-AMDGCN

! CHECK-AMDGCN: "-fc1" "-triple" "amdgcn-amd-amdhsa"
! CHECK-AMDGCN-SAME: "-target-cpu" "gfx902"
