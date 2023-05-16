// REQUIRES: amdgpu-registered-target
// RUN: %clang -E -dM --offload-arch=gfx906 -mwavefrontsize64 \
// RUN:   --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// RUN: %clang -E -dM --offload-arch=gfx1010 -mwavefrontsize64 \
// RUN:   --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// RUN: %clang -E -dM --offload-arch=gfx906 -mwavefrontsize64 \
// RUN:   --cuda-device-only -nogpuinc -nogpulib \
// RUN:   -mno-wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// RUN: %clang -E -dM --offload-arch=gfx1010 -mwavefrontsize64 \
// RUN:   --cuda-device-only -nogpuinc -nogpulib \
// RUN:   -mno-wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=WAVE32 %s
// RUN: %clang -E -dM --offload-arch=gfx906 -mno-wavefrontsize64 \
// RUN:   --cuda-device-only -nogpuinc -nogpulib \
// RUN:   -mwavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// RUN: %clang -E -dM --offload-arch=gfx1010 -mno-wavefrontsize64 \
// RUN:   --cuda-device-only -nogpuinc -nogpulib \
// RUN:   -mwavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// WAVE64-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// WAVE32-DAG: #define __AMDGCN_WAVEFRONT_SIZE 32

// RUN: %clang -E -dM --offload-arch=gfx906 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefix=CUMODE-ON %s
// RUN: %clang -E -dM --offload-arch=gfx906 --cuda-device-only -nogpuinc -nogpulib -mcumode \
// RUN:   %s 2>&1 | FileCheck --check-prefix=CUMODE-ON %s
// RUN: %clang -E -dM --offload-arch=gfx906 --cuda-device-only -nogpuinc -nogpulib -mno-cumode \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=CUMODE-ON,WARN-CUMODE %s
// RUN: %clang -E -dM --offload-arch=gfx1030 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefix=CUMODE-OFF %s
// RUN: %clang -E -dM --offload-arch=gfx1030 --cuda-device-only -nogpuinc -nogpulib -mcumode \
// RUN:   %s 2>&1 | FileCheck --check-prefix=CUMODE-ON %s
// RUN: %clang -E -dM --offload-arch=gfx1030 --cuda-device-only -nogpuinc -nogpulib -mno-cumode \
// RUN:   %s 2>&1 | FileCheck --check-prefix=CUMODE-OFF %s

// Check no duplicate warnings.
// RUN: %clang -E -dM --offload-arch=gfx906 --cuda-device-only -nogpuinc -nogpulib -mcumode \
// RUN:   -mno-cumode -mno-cumode \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=CUMODE-ON,WARN-CUMODE %s

// WARN-CUMODE-DAG: warning: ignoring '-mno-cumode' option as it is not currently supported for processor 'gfx906' [-Woption-ignored]
// WARN-CUMODE-NOT: warning: ignoring '-mno-cumode' option as it is not currently supported for processor 'gfx906' [-Woption-ignored]
// CUMODE-ON-DAG: #define __AMDGCN_CUMODE__ 1
// CUMODE-OFF-DAG: #define __AMDGCN_CUMODE__ 0
