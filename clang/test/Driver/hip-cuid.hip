// Check invalid -fuse-cuid= option.

// RUN: not %clang -### -x hip \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --no-offload-new-driver \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpuinc -nogpulib -fuse-cuid=invalid \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=INVALID %s

// Check random CUID generator.

// RUN: %clang -### -x hip \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --no-offload-new-driver \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpuinc -nogpulib -fuse-cuid=random \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,HEX %s

// Check fixed CUID.

// RUN: %clang -### -x hip \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --no-offload-new-driver \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpuinc -nogpulib -cuid=xyz_123 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,FIXED %s

// Check fixed CUID override -fuse-cuid.

// RUN: %clang -### -x hip \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --no-offload-new-driver \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpuinc -nogpulib -fuse-cuid=random -cuid=xyz_123 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,FIXED %s

// Check hash CUID generator.

// RUN: %clang -### -x hip \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --no-offload-new-driver \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpuinc -nogpulib -fuse-cuid=hash \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,HEX %s

// INVALID: invalid value 'invalid' in '-fuse-cuid=invalid'

// COMMON: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-target-cpu" "gfx900"
// HEX-SAME: "-cuid=[[CUID:[0-9a-f]+]]"
// FIXED-SAME: "-cuid=[[CUID:xyz_123]]"
// COMMON-SAME: "{{.*}}a.cu"

// COMMON: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-target-cpu" "gfx906"
// COMMON-SAME: "-cuid=[[CUID]]"
// COMMON-SAME: "{{.*}}a.cu"

// COMMON: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu"
// COMMON-SAME: "-cuid=[[CUID]]"
// COMMON-SAME: "{{.*}}a.cu"

// COMMON: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-target-cpu" "gfx900"
// HEX-NOT: "-cuid=[[CUID]]"
// HEX-SAME: "-cuid=[[CUID2:[0-9a-f]+]]"
// FIXED-SAME: "-cuid=[[CUID2:xyz_123]]"
// COMMON-SAME: "{{.*}}b.hip"

// COMMON: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-target-cpu" "gfx906"
// HEX-NOT: "-cuid=[[CUID]]"
// COMMON-SAME: "-cuid=[[CUID2]]"
// COMMON-SAME: "{{.*}}b.hip"

// COMMON: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu"
// HEX-NOT: "-cuid=[[CUID]]"
// COMMON-SAME: "-cuid=[[CUID2]]"
// COMMON-SAME: "{{.*}}b.hip"
