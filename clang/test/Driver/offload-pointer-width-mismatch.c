// Offload targets that take their pointer related types from the host target
// are rejected early, before any compilation job is built.

// RUN: not %clang -### --target=i386-unknown-linux-gnu -x hip --offload-arch=gfx906 \
// RUN:   -nogpulib -nogpuinc -c %s 2>&1 | FileCheck --check-prefix=AMDGPU %s
// RUN: not %clang -### --target=i386-unknown-linux-gnu -x cuda --offload=nvptx64-nvidia-cuda \
// RUN:   --offload-arch=sm_60 -nogpulib -nogpuinc -c %s 2>&1 | FileCheck --check-prefix=NVPTX %s
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -x cuda --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --offload=spirv32-unknown-unknown -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=SPIRV32 %s
// RUN: not %clang -### --target=i386-unknown-linux-gnu -x cuda \
// RUN:   --offload-targets=nvptx64-nvidia-cuda --offload-arch=sm_60 -nogpulib -nogpuinc \
// RUN:   -c %s 2>&1 | FileCheck --check-prefix=NVPTX %s

// OpenMP offloading reaches the same check.

// RUN: not %clang -### --target=i386-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -nogpulib -nogpuinc -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NVPTX %s
// RUN: not %clang -### --target=i386-unknown-linux-gnu -fopenmp --offload-arch=gfx906 \
// RUN:   -nogpulib -nogpuinc -c %s 2>&1 | FileCheck --check-prefix=AMDGPU %s

// AMDGPU: error: device target 'amdgpu-amd-amdhsa' takes a pointer width of 32 bits from host target 'i386-unknown-linux-gnu', but requires 64 bits
// NVPTX: error: device target 'nvptx64-nvidia-cuda' takes a pointer width of 32 bits from host target 'i386-unknown-linux-gnu', but requires 64 bits
// SPIRV32: error: device target 'spirv32-unknown-unknown' takes a pointer width of 64 bits from host target 'x86_64-unknown-linux-gnu', but requires 32 bits

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -x hip --offload-arch=gfx906 \
// RUN:   -nogpulib -nogpuinc -c %s 2>&1 | FileCheck --check-prefix=OK %s

// OK-NOT: error:
// OK: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
