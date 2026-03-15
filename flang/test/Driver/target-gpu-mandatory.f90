! REQUIRES: amdgpu-registered-target

! Test that -foffload-mandatory is accepted

! RUN: %flang --target=amdgcn-amd-amdhsa -mcpu=gfx902 -fopenmp-offload-mandatory  -nogpulib -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-MANDO
! CHECK-MANDO: "gfx902"
