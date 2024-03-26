// RUN: not %clang -### --target=x86_64-linux-gnu --no-offload-new-driver \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -c -fhip-emit-relocatable -nogpuinc -nogpulib --cuda-device-only -fgpu-rdc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=RELOCRDC %s

// RELOCRDC: error: option '-fhip-emit-relocatable' cannot be specified with '-fgpu-rdc'

// RUN: not %clang -### --target=x86_64-linux-gnu --no-offload-new-driver \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -c -fhip-emit-relocatable -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=RELOCHOST %s

// RELOCHOST: error: option '-fhip-emit-relocatable' cannot be specified without '--cuda-device-only'
