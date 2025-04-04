! REQUIRES: x86-registered-target, amdgpu-registered-target
! RUN: not %flang  -target amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=3  -S  %s -o \
! RUN:   /dev/null 2>&1 | FileCheck  --check-prefix=INVALID_VERSION %s

! RUN: %flang  -target  x86_64-unknown-linux-gnu -mcode-object-version=3  -S  %s -o \
! RUN:   /dev/null 2>&1 | FileCheck  --check-prefix=UNUSED_PARAM %s

! INVALID_VERSION: error: invalid integral value '3' in '-mcode-object-version=3'
! UNUSED_PARAM: warning: argument unused during compilation: '-mcode-object-version=3' [-Wunused-command-line-argument]
