// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: shell

// Check that -fsplit-machine-functions is passed to both x86 and cuda
// compilation and does not cause driver error.
// RUN:   cd "$(dirname "%t")" ; \
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions -S %s \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS1
// MFS1: "-target-cpu" "x86-64"{{.*}}"-fsplit-machine-functions"
// MFS1: "-target-cpu" "sm_70"{{.*}}"-fsplit-machine-functions"

// Check that -fsplit-machine-functions is passed to cuda and it
// causes a warning.
// RUN:   cd "$(dirname "%t")" ; \
// RUN:   %clang --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions -S %s \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS2
// MFS2: warning: -fsplit-machine-functions is not valid for nvptx

// Check that -Xarch_host -fsplit-machine-functions is passed only to
// native compilation.
// RUN:   cd "$(dirname "%t")" ; \
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -Xarch_host \
// RUN:     -fsplit-machine-functions -S %s \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS3
// MFS3:     "-target-cpu" "x86-64"{{.*}}"-fsplit-machine-functions"
// MFS3-NOT: "-target-cpu" "sm_70"{{.*}}"-fsplit-machine-functions"

// Check that -Xarch_host -fsplit-machine-functions does not cause any warning.
// RUN:   cd "$(dirname "%t")" ; \
// RUN:   %clang --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN      --cuda-gpu-arch=sm_70 -x cuda -Xarch_host \
// RUN      -fsplit-machine-functions -S %s || { echo \
// RUN      "warning: -fsplit-machine-functions is not valid for" ; } \
// RUN      2>&1 | FileCheck %s --check-prefix=MFS4
// MFS4-NOT: warning: -fsplit-machine-functions is not valid for

// Check that -Xarch_device -fsplit-machine-functions does cause the warning.
// RUN:   cd "$(dirname "%t")" ; \
// RUN:   %clang --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -Xarch_device \
// RUN:     -fsplit-machine-functions -S %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=MFS5
// MFS5: warning: -fsplit-machine-functions is not valid for

// Check that -fsplit-machine-functions -Xarch_device
// -fno-split-machine-functions only passes MFS to x86
// RUN:   cd "$(dirname "%t")" ; \
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions \
// RUN:     -Xarch_device -fno-split-machine-functions -S %s \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS6
// MFS6:     "-target-cpu" "x86-64"{{.*}}"-fsplit-machine-functions"
// MFS6-NOT: "-target-cpu" "sm_70"{{.*}}"-fsplit-machine-functions"

// Check that -fsplit-machine-functions -Xarch_device
// -fno-split-machine-functions has no warnings
// RUN:   cd "$(dirname "%t")" ; \
// RUN:   %clang --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions \
// RUN:     -Xarch_device -fno-split-machine-functions -S %s \
// RUN:     || { echo "warning: -fsplit-machine-functions is not valid for"; } \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS7
// MFS7-NOT: warning: -fsplit-machine-functions is not valid for
