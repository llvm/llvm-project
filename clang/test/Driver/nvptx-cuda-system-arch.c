// REQUIRES: system-linux
// REQUIRES: shell

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/nvptx-arch/nvptx_arch_fail %t/
// RUN: cp %S/Inputs/nvptx-arch/nvptx_arch_sm_70 %t/
// RUN: cp %S/Inputs/nvptx-arch/nvptx_arch_sm_89_sm_80 %t/
// RUN: echo '#!/bin/sh' > %t/nvptx_arch_empty
// RUN: chmod +x %t/nvptx_arch_fail
// RUN: chmod +x %t/nvptx_arch_sm_70
// RUN: chmod +x %t/nvptx_arch_sm_89_sm_80
// RUN: chmod +x %t/nvptx_arch_empty

// case when nvptx-arch returns nothing or fails
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --nvptx-arch-tool=%t/nvptx_arch_fail -x cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-new-driver --offload-arch=native --nvptx-arch-tool=%t/nvptx_arch_fail -x cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// NO-OUTPUT-ERROR: error: cannot determine nvptx64 architecture{{.*}}; consider passing it via '--offload-arch'

// case when nvptx-arch does not return anything with successful execution
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --nvptx-arch-tool=%t/nvptx_arch_empty -x cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-OUTPUT
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-new-driver --offload-arch=native --nvptx-arch-tool=%t/nvptx_arch_empty -x cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-OUTPUT
// EMPTY-OUTPUT: error: cannot determine nvptx64 architecture: No NVIDIA GPU detected in the system; consider passing it via '--offload-arch'

// case when nvptx-arch does not return anything with successful execution
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --nvptx-arch-tool=%t/nvptx_arch_sm_70 -x cuda --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-sm_70
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --offload-new-driver --nvptx-arch-tool=%t/nvptx_arch_sm_70 -x cuda --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-sm_70
// ARCH-sm_70: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_70"

// case when nvptx-arch is used via '-march=native'
// RUN:   %clang -### --target=nvptx64-nvidia-cuda -nogpulib -march=native --nvptx-arch-tool=%t/nvptx_arch_sm_70 \
// RUN:     --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda %s 2>&1 | FileCheck %s --check-prefix=MARCH-sm_70
// MARCH-sm_70: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_70"

// case when nvptx-arch is used via '-march=native'
// RUN:   %clang -### --target=nvptx64-nvidia-cuda -nogpulib -march=native --nvptx-arch-tool=%t/nvptx_arch_sm_89_sm_80 \
// RUN:     --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda %s 2>&1 | FileCheck %s --check-prefix=MARCH-sm_89
// MARCH-sm_89: warning: multiple nvptx64 architectures are detected: sm_89, sm_80; only the first one is used for '-march' [-Wmulti-gpu]
// MARCH-sm_89: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_89"
