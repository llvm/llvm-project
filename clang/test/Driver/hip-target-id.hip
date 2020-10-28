// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip \
// RUN:   --offload-arch=gfx908:xnack+:sram-ecc+ \
// RUN:   --offload-arch=gfx908:xnack+:sram-ecc- \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip \
// RUN:   --offload-arch=gfx908:xnack+:sram-ecc+ \
// RUN:   --offload-arch=gfx908:xnack+:sram-ecc- \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   -save-temps \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=CHECK,TMP %s

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip \
// RUN:   --offload-arch=gfx908:xnack+:sram-ecc+ \
// RUN:   --offload-arch=gfx908:xnack+:sram-ecc- \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   -fgpu-rdc \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=CHECK %s

// CHECK: [[CLANG:"[^"]*clang[^"]*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-target-cpu" "gfx908"
// CHECK-SAME: "-target-feature" "+sram-ecc"
// CHECK-SAME: "-target-feature" "+xnack"

// TMP: [[CLANG:"[^"]*clang[^"]*"]] "-cc1as" "-triple" "amdgcn-amd-amdhsa"
// TMP-SAME: "-target-cpu" "gfx908"
// TMP-SAME: "-target-feature" "+sram-ecc"
// TMP-SAME: "-target-feature" "+xnack"

// CHECK: [[LLD:"[^"]*lld[^"]*"]] {{.*}} "-plugin-opt=mcpu=gfx908"
// CHECK-SAME: "-plugin-opt=-mattr=+sram-ecc,+xnack"

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-target-cpu" "gfx908"
// CHECK-SAME: "-target-feature" "-sram-ecc"
// CHECK-SAME: "-target-feature" "+xnack"

// CHECK: [[LLD]] {{.*}} "-plugin-opt=mcpu=gfx908"
// CHECK-SAME: "-plugin-opt=-mattr=-sram-ecc,+xnack"

// CHECK: {{"[^"]*clang-offload-bundler[^"]*"}}
// CHECK-SAME: "-targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-gfx908:sram-ecc+:xnack+,hip-amdgcn-amd-amdhsa-gfx908:sram-ecc-:xnack+"

// Check canonicalization and repeating of target ID.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip \
// RUN:   --offload-arch=fiji \
// RUN:   --offload-arch=gfx803 \
// RUN:   --offload-arch=fiji \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=FIJI %s
// FIJI: "-targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-gfx803"

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip \
// RUN:   --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ \
// RUN:   --offload-arch=gfx908:sram-ecc+ \
// RUN:   --offload-arch=gfx908:sram-ecc- \
// RUN:   --offload-arch=gfx906 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=MULTI %s
// MULTI: "-targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-gfx900:xnack+,hip-amdgcn-amd-amdhsa-gfx900:xnack-,hip-amdgcn-amd-amdhsa-gfx906,hip-amdgcn-amd-amdhsa-gfx908:sram-ecc+,hip-amdgcn-amd-amdhsa-gfx908:sram-ecc-"
