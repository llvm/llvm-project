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
// WAVE64-DAG: #define __AMDGCN_WAVEFRONT_SIZE__ 64
// WAVE32-DAG: #define __AMDGCN_WAVEFRONT_SIZE__ 32
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

// RUN: %clang -E -dM --offload-arch=gfx90a --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=IMAGE,NOWARN %s
// RUN: %clang -E -dM --offload-arch=gfx1100 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=IMAGE,NOWARN %s
// RUN: %clang -E -dM --offload-arch=gfx940 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=NOIMAGE,NOWARN %s
// RUN: %clang -E -dM --offload-arch=gfx941 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=NOIMAGE,NOWARN %s
// RUN: %clang -E -dM --offload-arch=gfx942 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=NOIMAGE,NOWARN %s
// RUN: %clang -E -dM --offload-arch=gfx1100 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   -Xclang -target-feature -Xclang "-image-insts" %s 2>&1 | FileCheck --check-prefixes=IMAGE,WARN %s
// RUN: %clang -E -dM --offload-arch=gfx940 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   -Xclang -target-feature -Xclang "+image-insts" %s 2>&1 | FileCheck --check-prefixes=NOIMAGE,WARN %s
// NOWARN-NOT: warning
// WARN: warning: feature flag '{{[+|-]}}image-insts' is ignored since the feature is read only [-Winvalid-command-line-argument]
// IMAGE-NOT: #define __HIP_NO_IMAGE_SUPPORT__
// IMAGE-NOT: #define __HIP_NO_IMAGE_SUPPORT
// NOIMAGE-DAG: #define __HIP_NO_IMAGE_SUPPORT__ 1
// NOIMAGE-DAG: #define __HIP_NO_IMAGE_SUPPORT 1

// RUN: %clang -E -dM --offload-arch=gfx1100 -nogpuinc -nogpulib \
// RUN:   -fgpu-default-stream=per-thread %s 2>&1 | FileCheck --check-prefixes=PTS %s
// RUN: %clang -E -dM --offload-arch=gfx940 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   -fgpu-default-stream=legacy %s 2>&1 | FileCheck --check-prefixes=NOPTS %s
// RUN: %clang -E -dM --offload-arch=gfx940 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=NOPTS %s
// PTS-DAG: #define __HIP_API_PER_THREAD_DEFAULT_STREAM__ 1
// PTS-DAG: #define __HIP_API_PER_THREAD_DEFAULT_STREAM__ 1
// PTS-DAG: #define HIP_API_PER_THREAD_DEFAULT_STREAM 1
// PTS-DAG: #define HIP_API_PER_THREAD_DEFAULT_STREAM 1
// NOPTS-NOT: #define __HIP_API_PER_THREAD_DEFAULT_STREAM__
// NOPTS-NOT: #define HIP_API_PER_THREAD_DEFAULT_STREAM
