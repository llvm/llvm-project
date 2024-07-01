// Check that -fcf-protection does not get passed to the device-side
// compilation.

// RUN: %clang -### -x cuda --target=x86_64-unknown-linux-gnu -nogpulib \
// RUN:   -nogpuinc --offload-arch=sm_52 -fcf-protection=full -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CUDA

// CUDA: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDA-NOT: "-fcf-protection=full"
// CUDA: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CUDA: "-fcf-protection=full"

// RUN: %clang -### -x hip --target=x86_64-unknown-linux-gnu -nogpulib \
// RUN:   -nogpuinc --offload-arch=gfx90a -fcf-protection=full -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=HIP

// HIP: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// HIP-NOT: "-fcf-protection=full"
// HIP: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HIP: "-fcf-protection=full"

// RUN: %clang -### -x c --target=x86_64-unknown-linux-gnu -nogpulib -fopenmp=libomp \
// RUN:   -nogpuinc --offload-arch=gfx90a -fcf-protection=full -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=OMP

// OMP: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// OMP: "-fcf-protection=full"
// OMP: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// OMP-NOT: "-fcf-protection=full"
// OMP: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// OMP: "-fcf-protection=full"

// RUN: %clang -### -x c --target=nvptx64-nvidia-cuda -nogpulib -nogpuinc \
// RUN:   -march=sm_52 -fcf-protection=full -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=DIRECT
// RUN: %clang -### -x c --target=amdgcn-amd-amdhsa -nogpulib -nogpuinc \
// RUN:   -mcpu=gfx90a -fcf-protection=full -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=DIRECT
// DIRECT: "-fcf-protection=full"
