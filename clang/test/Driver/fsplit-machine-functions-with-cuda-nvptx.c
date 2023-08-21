// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Check that -fsplit-machine-functions is passed to both x86 and cuda
// compilation and does not cause driver error.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions -S %s \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS1
// MFS1: "-target-cpu" "x86-64"{{.*}}"-fsplit-machine-functions"
// MFS1: "-target-cpu" "sm_70"{{.*}}"-fsplit-machine-functions"

// Check that -Xarch_host -fsplit-machine-functions is passed only to
// native compilation.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -Xarch_host \
// RUN:     -fsplit-machine-functions -S %s \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS2
// MFS2:     "-target-cpu" "x86-64"{{.*}}"-fsplit-machine-functions"
// MFS2-NOT: "-target-cpu" "sm_70"{{.*}}"-fsplit-machine-functions"

// Check that -fsplit-machine-functions -Xarch_device
// -fno-split-machine-functions only passes MFS to x86
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions \
// RUN:     -Xarch_device -fno-split-machine-functions -S %s \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS3
// MFS3:     "-target-cpu" "x86-64"{{.*}}"-fsplit-machine-functions"
// MFS3-NOT: "-target-cpu" "sm_70"{{.*}}"-fsplit-machine-functions"
