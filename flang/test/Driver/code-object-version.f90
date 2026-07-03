! REQUIRES: x86-registered-target, amdgpu-registered-target
! RUN: not %flang  -target amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=3  -S  %s -o \
! RUN:   /dev/null 2>&1 | FileCheck  --check-prefix=INVALID_VERSION %s

! RUN: %flang  -target  x86_64-unknown-linux-gnu -mcode-object-version=3  -S  %s -o \
! RUN:   /dev/null 2>&1 | FileCheck  --check-prefix=UNUSED_PARAM %s

! RUN: %flang -target amdgcn-amd-amdhsa -mcpu=gfx908 -mcode-object-version=5 -nogpulib -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=VALID_USE

! INVALID_VERSION: error: invalid integral value '3' in '-mcode-object-version=3'
! UNUSED_PARAM: warning: argument unused during compilation: '-mcode-object-version=3' [-Wunused-command-line-argument]

! VALID_USE: "-fc1" "-triple" "amdgcn-amd-amdhsa"
! VALID_USE-SAME: "-mcode-object-version=5"
! VALID_USE-SAME: "-mllvm" "--amdhsa-code-object-version=5"
