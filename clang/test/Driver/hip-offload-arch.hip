// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx1030 \
// RUN:   --offload-arch=gfx1031 \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck %s


// CHECK: "-cc1" "-triple" "amdgpu10.30-amd-amdhsa"{{.*}}"-target-cpu" "gfx1030"
// CHECK: "-cc1" "-triple" "amdgpu10.31-amd-amdhsa"{{.*}}"-target-cpu" "gfx1031"

// RUN: %clang -### --target=x86_64-linux-gnu --offload-targets=amdgpu9-amd-amdhsa \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=SUBARCH-TRIPLE %s

// SUBARCH-TRIPLE-NOT: "-cc1"
// SUBARCH-TRIPLE: "-cc1" "-triple" "amdgpu9.00-amd-amdhsa"{{.*}}"-target-cpu" "gfx900"
// SUBARCH-TRIPLE-NEXT: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"{{.*}}"-target-cpu" "gfx906"
// SUBARCH-TRIPLE-NOT: "-cc1"
// SUBARCH-TRIPLE: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// SUBARCH-TRIPLE-NOT: "-cc1"

// gfx90a is its own major subarch, so it is not covered by the amdgpu9 target.
// RUN: not %clang -### --target=x86_64-linux-gnu --offload-targets=amdgpu9-amd-amdhsa \
// RUN:   --offload-arch=gfx90a \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=GFX90A-NOT-IN-GFX9 %s

// GFX90A-NOT-IN-GFX9: clang: error: the target architecture 'gfx90a' is not supported by the target 'amdgpu9'

// RUN: not %clang -### --target=x86_64-linux-gnu --offload-targets=amdgpu12-amd-amdhsa \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx90a \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=OTHER-SUBARCH-TRIPLE-ERR %s

// OTHER-SUBARCH-TRIPLE-ERR: clang: error: the target architecture 'gfx900' is not supported by the target 'amdgpu12'
