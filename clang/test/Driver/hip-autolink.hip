// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
//
// RUN: %clang --target=i386-pc-windows-msvc --cuda-gpu-arch=gfx906 -nogpulib \
// RUN:   --cuda-device-only %s -### 2>&1 | FileCheck --check-prefix=DEV %s
// RUN: %clang --target=i386-pc-windows-msvc --cuda-gpu-arch=gfx906 -nogpulib \
// RUN:   --cuda-host-only %s -### 2>&1 | FileCheck --check-prefix=HOST %s

// DEV: "-cc1" "-mllvm" "--amdhsa-code-object-version=4" "-triple" "amdgcn-amd-amdhsa"
// DEV-SAME: "-fno-autolink"

// HOST: "-cc1" "-triple" "i386-pc-windows-msvc{{.*}}"
// HOST-NOT: "-fno-autolink"
