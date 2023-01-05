// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: shell

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/nvptx-arch/nvptx_arch_fail %t/
// RUN: cp %S/Inputs/nvptx-arch/nvptx_arch_sm_70 %t/
// RUN: echo '#!/bin/sh' > %t/nvptx_arch_empty
// RUN: chmod +x %t/nvptx_arch_fail
// RUN: chmod +x %t/nvptx_arch_sm_70
// RUN: chmod +x %t/nvptx_arch_empty

// case when nvptx-arch returns nothing or fails
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --nvptx-arch-tool=%t/nvptx_arch_fail -x cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// NO-OUTPUT-ERROR: error: cannot determine nvptx64 architecture{{.*}}; consider passing it via '--offload-arch'

// case when nvptx-arch does not return anything with successful execution
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --nvptx-arch-tool=%t/nvptx_arch_empty -x cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-OUTPUT
// EMPTY-OUTPUT: error: cannot determine nvptx64 architecture: No NVIDIA GPU detected in the system; consider passing it via '--offload-arch'

// case when nvptx-arch does not return anything with successful execution
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --nvptx-arch-tool=%t/nvptx_arch_sm_70 -x cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-sm_70
// ARCH-sm_70: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_70"
