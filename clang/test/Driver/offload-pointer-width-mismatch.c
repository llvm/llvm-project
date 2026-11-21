// An offload compilation whose device target takes its pointer related types
// from an incompatible host target fails when the device compilation runs.

// REQUIRES: amdgpu-registered-target, nvptx-registered-target

// RUN: not %clang --target=i386-unknown-linux-gnu -x hip --offload-arch=gfx906 \
// RUN:   -nogpulib -nogpuinc -fsyntax-only %s 2>&1 | FileCheck --check-prefix=AMDGPU %s
// RUN: not %clang --target=i386-unknown-linux-gnu -x cuda --offload=nvptx64-nvidia-cuda \
// RUN:   --offload-arch=sm_60 -nogpulib -nogpuinc -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NVPTX %s
// RUN: not %clang --target=i386-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_60 -nogpulib -nogpuinc \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=NVPTX %s

// AMDGPU: error: device target 'amdgpu9.06-amd-amdhsa' is not compatible with host target 'i386-unknown-linux-gnu'
// AMDGPU-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// NVPTX: error: device target 'nvptx64-nvidia-cuda' is not compatible with host target 'i386-unknown-linux-gnu'
// NVPTX-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)

// A host target of a matching pointer width is accepted, and so is the 32-bit
// device target that a 32-bit host selects by default.

// RUN: %clang --target=x86_64-unknown-linux-gnu -x hip --offload-arch=gfx906 \
// RUN:   -nogpulib -nogpuinc -fsyntax-only %s
// RUN: %clang --target=i386-unknown-linux-gnu -x cuda --offload-arch=sm_60 \
// RUN:   -nogpulib -nogpuinc -fsyntax-only %s
