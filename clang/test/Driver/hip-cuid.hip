// Check cuid is supported.
// RUN: %clang -### -x hip \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,HEX %s

// Check cuid is supported for CUDA.
// RUN: %clang -### -x cu \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=sm_60 \
// RUN:   --offload-arch=sm_70 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,NVPTX,HEX %s

// COMMON: "-cc1"{{.*}} "-triple" "[[TRIP0:(amdgpu9.00-amd-amdhsa|nvptx64-nvidia-cuda)]]"
// NVPTX-SAME: "-target-cpu" "[[G1:sm_60]]"
// HEX-SAME: "-cuid=[[CUID:[0-9a-f]+]]"
// COMMON-SAME: "{{.*}}a.cu"

// COMMON: "-cc1"{{.*}} "-triple" "[[TRIP1:(amdgpu9.06-amd-amdhsa|nvptx64-nvidia-cuda)]]"
// NVPTX-SAME: "-target-cpu" "[[G2:sm_70]]"
// COMMON-SAME: "-cuid=[[CUID]]"
// COMMON-SAME: "{{.*}}a.cu"

// COMMON: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu"
// COMMON-SAME: "-cuid=[[CUID]]"
// COMMON-SAME: "{{.*}}a.cu"

// COMMON: "-cc1"{{.*}} "-triple" "[[TRIP0]]"
// NVPTX-SAME: "-target-cpu" "[[G1]]"
// HEX-NOT: "-cuid=[[CUID]]"
// HEX-SAME: "-cuid=[[CUID2:[0-9a-f]+]]"
// COMMON-SAME: "{{.*}}b.hip"

// COMMON: "-cc1"{{.*}} "-triple" "[[TRIP1]]"
// NVPTX-SAME: "-target-cpu" "[[G2]]"
// HEX-NOT: "-cuid=[[CUID]]"
// COMMON-SAME: "-cuid=[[CUID2]]"
// COMMON-SAME: "{{.*}}b.hip"

// COMMON: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu"
// HEX-NOT: "-cuid=[[CUID]]"
// COMMON-SAME: "-cuid=[[CUID2]]"
// COMMON-SAME: "{{.*}}b.hip"
