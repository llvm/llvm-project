// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -E -dM -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx908:xnack+:sram-ecc- -nogpulib -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=PROC,ID1 %s

// RUN: %clang -E -dM -target amdgcn-amd-amdpal \
// RUN:   -mcpu=gfx908:xnack+:sram-ecc- -nogpulib -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=PROC,ID1 %s

// RUN: %clang -E -dM -target amdgcn--mesa3d \
// RUN:   -mcpu=gfx908:xnack+:sram-ecc- -nogpulib -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=PROC,ID1 %s

// RUN: %clang -E -dM -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx908 -nogpulib -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=PROC,ID2 %s

// RUN: %clang -E -dM -target amdgcn-amd-amdhsa \
// RUN:   -nogpulib -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NONE %s

// PROC-DAG: #define __amdgcn_processor__ "gfx908"

// ID1-DAG: #define __amdgcn_feature_xnack__ 1
// ID1-DAG: #define __amdgcn_feature_sram_ecc__ 0
// ID1-DAG: #define __amdgcn_target_id__ "gfx908:sram-ecc-:xnack+"

// ID2-DAG: #define __amdgcn_target_id__ "gfx908"
// ID2-NOT: #define __amdgcn_feature_xnack__
// ID2-NOT: #define __amdgcn_feature_sram_ecc__

// NONE-NOT: #define __amdgcn_processor__
// NONE-NOT: #define __amdgcn_feature_xnack__
// NONE-NOT: #define __amdgcn_feature_sram_ecc__
// NONE-NOT: #define __amdgcn_target_id__
