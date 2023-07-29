// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -c --target=x86_64-linux-gnu -march=znver2 --cuda-gpu-arch=gfx803 -nogpulib %s 2>&1 | FileCheck %s -check-prefix=HOSTCPU
// RUN: %clang -### -c --target=x86_64-linux-gnu -msse3 --cuda-gpu-arch=gfx803 -nogpulib %s 2>&1 | FileCheck %s -check-prefix=HOSTSSE3
// RUN: %clang -### -c --target=x86_64-linux-gnu --gpu-use-aux-triple-only -march=znver2 --cuda-gpu-arch=gfx803 -nogpulib %s 2>&1 | FileCheck %s -check-prefix=NOHOSTCPU

// HOSTCPU: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// HOSTCPU-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// HOSTCPU-SAME: "-aux-target-cpu" "znver2"

// HOSTSSE3: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// HOSTSSE3-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// HOSTSSE3-SAME: "-aux-target-feature" "+sse3"

// NOHOSTCPU: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// NOHOSTCPU-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// NOHOSTCPU-NOT: "-aux-target-cpu" "znver2"
