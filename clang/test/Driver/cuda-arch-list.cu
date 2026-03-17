// Checks that __CUDA_ARCH_LIST__ is defined correctly for both host and device
// subcompilations.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nocudainc -nocudalib \
// RUN:   --offload-arch=sm_60 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=DEVICE60,HOST %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nocudainc -nocudalib \
// RUN:   --offload-arch=sm_60 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=DEVICE60-60-70,DEVICE70-60-70,HOST-60-70 %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nocudainc -nocudalib \
// RUN:   --offload-arch=sm_70 --offload-arch=sm_60 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=DEVICE60-60-70,DEVICE70-60-70,HOST-60-70 %s

// Verify that it works with no explicit arch (defaults to sm_52)
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nocudainc -nocudalib \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=DEVICE52,HOST52 %s

// Verify that --no-offload-arch negates preceding --offload-arch
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nocudainc -nocudalib \
// RUN:   --offload-arch=sm_60 --offload-arch=sm_70 --no-offload-arch=sm_60 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=DEVICE70-ONLY,HOST70-ONLY %s

// Verify that user-specified -D__CUDA_ARCH_LIST__ overrides the driver-generated one
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nocudainc -nocudalib \
// RUN:   --offload-arch=sm_60 -D__CUDA_ARCH_LIST__=999 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=DEVICE-OVERRIDE,HOST-OVERRIDE %s

// DEVICE60: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE60-SAME: "-target-cpu" "sm_60"
// DEVICE60-SAME: "-D__CUDA_ARCH_LIST__=600"

// HOST: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-SAME: "-D__CUDA_ARCH_LIST__=600"

// DEVICE60-60-70: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE60-60-70-SAME: "-target-cpu" "sm_60"
// DEVICE60-60-70-SAME: "-D__CUDA_ARCH_LIST__=600,700"

// DEVICE70-60-70: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE70-60-70-SAME: "-target-cpu" "sm_70"
// DEVICE70-60-70-SAME: "-D__CUDA_ARCH_LIST__=600,700"

// HOST-60-70: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-60-70-SAME: "-D__CUDA_ARCH_LIST__=600,700"

// DEVICE52: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE52-SAME: "-target-cpu" "sm_52"
// DEVICE52-SAME: "-D__CUDA_ARCH_LIST__=520"

// HOST52: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST52-SAME: "-D__CUDA_ARCH_LIST__=520"

// DEVICE70-ONLY: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE70-ONLY-SAME: "-target-cpu" "sm_70"
// DEVICE70-ONLY-SAME: "-D__CUDA_ARCH_LIST__=700"

// HOST70-ONLY: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST70-ONLY-SAME: "-D__CUDA_ARCH_LIST__=700"

// DEVICE-OVERRIDE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-OVERRIDE-SAME: "-target-cpu" "sm_60"
// DEVICE-OVERRIDE-SAME: "-D__CUDA_ARCH_LIST__=600"
// DEVICE-OVERRIDE-SAME: "-D" "__CUDA_ARCH_LIST__=999"

// HOST-OVERRIDE: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-OVERRIDE-SAME: "-D__CUDA_ARCH_LIST__=600"
// HOST-OVERRIDE-SAME: "-D" "__CUDA_ARCH_LIST__=999"
